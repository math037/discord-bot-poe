import os
import aiohttp
import discord
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
HF_API_KEY = os.environ["HF_API_KEY"]

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Discord message length limit
DISCORD_MAX_LENGTH = 2000

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


async def get_hf_response(user_message: str) -> str:
    """Call the Hugging Face Inference API and return the generated text."""
    # Mistral instruct format
    prompt = f"<s>[INST] {user_message} [/INST]"
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512,
            "temperature": 0.7,
            "return_full_text": False,
        },
    }
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/json",
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(
            HF_API_URL,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=60),
        ) as resp:
            if resp.status == 503:
                # Model is loading — surface a friendly message
                raise RuntimeError(
                    "The AI model is currently loading. Please try again in a moment."
                )
            if resp.status != 200:
                body = await resp.text()
                raise RuntimeError(
                    f"Hugging Face API returned HTTP {resp.status}: {body[:200]}"
                )

            data = await resp.json()

            # Response is a list of dicts: [{"generated_text": "..."}]
            if isinstance(data, list) and data:
                return data[0].get("generated_text", "").strip()

            raise RuntimeError(f"Unexpected response format: {str(data)[:200]}")


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
    print("Using Hugging Face Inference API (mistralai/Mistral-7B-Instruct-v0.1)")


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
            reply = await get_hf_response(content)
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
            print(f"Hugging Face API error: {exc}")
            await message.reply(f"⚠️ {exc}")
        except aiohttp.ClientError as exc:
            print(f"HTTP error calling Hugging Face API: {exc}")
            await message.reply("⚠️ Could not reach the AI API. Please try again later.")
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            await message.reply("⚠️ Something went wrong. Please try again later.")


client.run(DISCORD_TOKEN)


