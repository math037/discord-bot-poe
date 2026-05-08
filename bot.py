import asyncio
import logging
import os
import aiohttp
import discord
from discord.errors import LoginFailure
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

HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"

# Discord message length limit
DISCORD_MAX_LENGTH = 2000

intents = discord.Intents.default()
intents.message_content = True

# Module-level session reference — set once in main() before the bot starts.
# All HF API calls share this single session for the lifetime of the process.
_session: aiohttp.ClientSession | None = None


async def get_hf_response(user_message: str) -> str:
    """Call the Hugging Face Inference API and return the generated text.

    Reuses the module-level ``_session`` that was created at startup so that
    the same underlying TCP connection pool is used for every request and no
    new session is opened (or closed) per call.
    """
    if _session is None or _session.closed:
        raise RuntimeError("HTTP session is not available.")

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

    async with _session.post(
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


def make_client(connector: aiohttp.TCPConnector) -> discord.Client:
    """Create and return a Discord client wired to the shared connector."""
    _client = discord.Client(intents=intents, connector=connector)

    @_client.event
    async def on_ready():
        logger.info(f"Logged in as {_client.user} (ID: {_client.user.id})")
        logger.info(
            "Using Hugging Face Inference API (mistralai/Mistral-7B-Instruct-v0.1)"
        )

    @_client.event
    async def on_message(message: discord.Message):
        # Ignore messages sent by the bot itself
        if message.author == _client.user:
            return

        # Only respond to @mentions and DMs
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = _client.user in message.mentions
        if not is_dm and not is_mentioned:
            return

        # Strip the bot mention from the message content
        content = message.content
        if is_mentioned:
            content = (
                content
                .replace(f"<@{_client.user.id}>", "")
                .replace(f"<@!{_client.user.id}>", "")
                .strip()
            )

        if not content:
            await message.reply("Hey! Ask me anything.")
            return

        async with message.channel.typing():
            try:
                reply = await get_hf_response(content)
                if not reply:
                    await message.reply(
                        "⚠️ Received an empty response. Please try again."
                    )
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
                await message.reply(
                    "⚠️ Could not reach the AI API. Please try again later."
                )
            except Exception as exc:
                logger.error(f"Unexpected error: {exc}")
                await message.reply(
                    "⚠️ Something went wrong. Please try again later."
                )

    return _client


async def main() -> None:
    """Entry point.

    Creates a single ``aiohttp.ClientSession`` (backed by a ``TCPConnector``)
    once, passes the connector to the Discord client so both share the same
    connection pool, then starts the bot.  discord.py's ``reconnect=True``
    (the default) handles all gateway-level reconnections internally, so no
    outer retry loop is needed — and no new session is ever created on
    reconnect.

    The session is closed in the ``finally`` block so it is always cleaned up
    on both clean shutdown and unexpected errors.
    """
    global _session

    # A single connector / session for the entire process lifetime.
    # connector_owner=False tells aiohttp.ClientSession not to close the
    # connector when the session itself is closed — we manage its lifetime here.
    connector = aiohttp.TCPConnector()
    _session = aiohttp.ClientSession(connector=connector, connector_owner=False)

    client = make_client(connector)

    try:
        logger.info("Starting Discord bot …")
        # reconnect=True (default) lets discord.py handle gateway drops,
        # RESUME packets, and backoff entirely on its own.
        await client.start(DISCORD_TOKEN, reconnect=True)
    except LoginFailure as exc:
        logger.critical("Invalid Discord token — cannot log in: %s", exc)
        raise
    except Exception as exc:
        logger.error("Bot exited with error: %s", exc)
        raise
    finally:
        logger.info("Shutting down — closing HTTP session …")
        if not _session.closed:
            await _session.close()
        if not connector.closed:
            await connector.close()
        logger.info("Shutdown complete.")


asyncio.run(main())

