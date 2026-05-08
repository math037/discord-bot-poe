import os
import aiohttp
import discord
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
POE_API_KEY = os.environ["POE_API_KEY"]

POE_API_URL = "https://api.poe.com/openai/v1/chat/completions"
POE_MODEL = "GPT-4o-Mini"

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


async def get_poe_response(user_message: str) -> str:
    """Send a message to the Poe OpenAI-compatible API and return the reply."""
    headers = {
        "Authorization": f"Bearer {POE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": POE_MODEL,
        "messages": [
            {"role": "user", "content": user_message},
        ],
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(POE_API_URL, json=payload, headers=headers) as resp:
            resp.raise_for_status()
            data = await resp.json()
            return data["choices"][0]["message"]["content"]


@client.event
async def on_ready():
    print(f"Logged in as {client.user} (ID: {client.user.id})")


@client.event
async def on_message(message: discord.Message):
    # Ignore messages sent by the bot itself
    if message.author == client.user:
        return

    # Ignore messages that don't mention the bot and aren't DMs
    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mentioned = client.user in message.mentions
    if not is_dm and not is_mentioned:
        return

    # Strip the bot mention from the message content, if present
    content = message.content
    if is_mentioned:
        content = content.replace(f"<@{client.user.id}>", "").replace(
            f"<@!{client.user.id}>", ""
        ).strip()

    if not content:
        await message.reply("Hey! Ask me anything.")
        return

    async with message.channel.typing():
        try:
            reply = await get_poe_response(content)
            await message.reply(reply)
        except aiohttp.ClientResponseError as exc:
            print(f"Poe API error {exc.status}: {exc.message}")
            await message.reply(
                f"⚠️ Poe API returned an error ({exc.status}). Please try again later."
            )
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            await message.reply("⚠️ Something went wrong. Please try again later.")


client.run(DISCORD_TOKEN)
