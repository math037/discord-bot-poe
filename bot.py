import os
import json
import aiohttp
import discord
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
POE_API_KEY = os.environ["POE_API_KEY"]

POE_BOT_NAME = os.environ.get("POE_BOT_NAME", "Claude-3.5-Sonnet")
POE_API_URL = "https://api.poe.com/api/query"

# Discord message length limit
DISCORD_MAX_LENGTH = 2000

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


async def get_poe_response(user_message: str) -> str:
    """Call the Poe HTTP API directly and return the full reply text."""
    payload = {
        "query": [{"role": "user", "content": user_message}],
        "bot_name": POE_BOT_NAME,
        "api_key": POE_API_KEY,
    }
    headers = {
        "Authorization": f"Bearer {POE_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            POE_API_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Poe API returned HTTP {resp.status}: {body[:200]}"
                )

            # The Poe query endpoint returns server-sent events (text/event-stream).
            # Each line is either "data: <json>" or blank/comment.
            reply_parts: list[str] = []
            async for raw_line in resp.content:
                line = raw_line.decode("utf-8").rstrip("\n")
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str.strip() in ("", "[DONE]"):
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("event")
                if event_type == "text":
                    reply_parts.append(event.get("data", {}).get("text", ""))
                elif event_type == "replace_response":
                    # Replace the entire accumulated response so far
                    reply_parts = [event.get("data", {}).get("text", "")]
                elif event_type == "error":
                    error_msg = event.get("data", {}).get("text", "Unknown error")
                    raise RuntimeError(f"Poe API error event: {error_msg}")
                elif event_type == "done":
                    break

            return "".join(reply_parts).strip()


def split_message(text: str, limit: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split a long message into chunks that fit within Discord's limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")
    print(f"Using Poe bot: {POE_BOT_NAME}")


@client.event
async def on_message(message: discord.Message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Only respond to @mentions and DMs
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions
    if not is_dm and not is_mentioned:
        return

    # Strip the bot mention from the message content
    content = message.content
    if is_mentioned:
        content = (
            content
            .replace(f"<@{client.user.id}>", "")
            .replace(f"<@!{client.user.id}>", "")
            .strip()
        )

    if not content:
        await message.reply("Hey! Ask me anything.")
        return

    async with message.channel.typing():
        try:
            reply = await get_poe_response(content)
            if not reply:
                await message.reply("⚠️ Received an empty response. Please try again.")
                return
            # Send reply in chunks if it exceeds Discord's 2000-char limit
            chunks = split_message(reply)
            for i, chunk in enumerate(chunks):
                if i == 0:
                    await message.reply(chunk)
                else:
                    await message.channel.send(chunk)
        except RuntimeError as exc:
            print(f"Poe API error: {exc}")
            await message.reply("⚠️ Poe API returned an error. Please try again later.")
        except aiohttp.ClientError as exc:
            print(f"HTTP error calling Poe API: {exc}")
            await message.reply("⚠️ Could not reach Poe API. Please try again later.")
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            await message.reply("⚠️ Something went wrong. Please try again later.")


client.run(DISCORD_TOKEN)

