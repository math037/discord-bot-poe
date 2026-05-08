import logging
import os
import time
import aiohttp
import discord
from discord.errors import HTTPException, LoginFailure
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
HF_API_KEY = os.environ["HF_API_KEY"]

# Reconnection settings
MAX_RETRIES = 5
BACKOFF_BASE = 2   # seconds — delay doubles each attempt (2, 4, 8, 16, 32)
BACKOFF_MAX = 60   # seconds — cap so we never wait longer than a minute

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
    logger.info(f"Logged in as {client.user} (ID: {client.user.id})")
    logger.info("Using Hugging Face Inference API (mistralai/Mistral-7B-Instruct-v0.1)")


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
            logger.error(f"Hugging Face API error: {exc}")
            await message.reply(f"⚠️ {exc}")
        except aiohttp.ClientError as exc:
            logger.error(f"HTTP error calling Hugging Face API: {exc}")
            await message.reply("⚠️ Could not reach the AI API. Please try again later.")
        except Exception as exc:
            logger.error(f"Unexpected error: {exc}")
            await message.reply("⚠️ Something went wrong. Please try again later.")


def run_bot() -> None:
    """Run the bot with exponential backoff on transient connection failures.

    discord.py's built-in ``reconnect=True`` handles short-lived gateway
    drops automatically.  This outer loop handles the harder cases: HTTP 429
    rate-limit responses on login and other unexpected errors that cause
    ``client.run()`` itself to raise.
    """
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            logger.info(
                "Starting Discord bot (attempt %d/%d) …", attempt + 1, MAX_RETRIES
            )
            # reconnect=True tells discord.py to handle gateway reconnections
            # internally (network blips, server restarts, etc.).
            client.run(DISCORD_TOKEN, reconnect=True)
            # client.run() returns normally only when the bot shuts down cleanly.
            logger.info("Bot shut down cleanly.")
            break
        except LoginFailure as exc:
            # Bad token — retrying will never help.
            logger.critical("Invalid Discord token — cannot log in: %s", exc)
            raise
        except HTTPException as exc:
            if exc.status == 429:
                retry_after = getattr(exc, "retry_after", None)
                if retry_after:
                    wait = min(float(retry_after), BACKOFF_MAX)
                    logger.warning(
                        "Discord rate-limited (429). Retrying after %.1fs "
                        "(as instructed by Discord) …",
                        wait,
                    )
                else:
                    wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                    logger.warning(
                        "Discord rate-limited (429). Retrying in %.1fs "
                        "(attempt %d/%d) …",
                        wait,
                        attempt + 1,
                        MAX_RETRIES,
                    )
                time.sleep(wait)
            else:
                wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
                logger.error(
                    "Discord HTTP error %d: %s. Retrying in %.1fs "
                    "(attempt %d/%d) …",
                    exc.status,
                    exc,
                    wait,
                    attempt + 1,
                    MAX_RETRIES,
                )
                time.sleep(wait)
        except Exception as exc:
            wait = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
            logger.error(
                "Unexpected error: %s. Retrying in %.1fs (attempt %d/%d) …",
                exc,
                wait,
                attempt + 1,
                MAX_RETRIES,
            )
            time.sleep(wait)

        attempt += 1

    if attempt >= MAX_RETRIES:
        logger.critical(
            "Bot failed to connect after %d attempts. Giving up.", MAX_RETRIES
        )
        raise SystemExit(1)


run_bot()
