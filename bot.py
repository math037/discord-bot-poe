import asyncio
import logging
import os

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

HF_API_URL = (
    "https://api-inference.huggingface.co/models/distilgpt2"
)

# Discord message length limit
DISCORD_MAX_LENGTH = 2000

# How long to wait (seconds) before retrying a loading model (503)
HF_MODEL_LOADING_RETRY_DELAY = 10
HF_MODEL_LOADING_MAX_RETRIES = 6


def split_message(text: str, limit: int = DISCORD_MAX_LENGTH) -> list[str]:
    """Split *text* into chunks that each fit within Discord's character limit."""
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


class DiscordBot(discord.Client):
    """A Discord bot that responds to @mentions with Hugging Face AI responses.

    Design principles
    -----------------
    * One ``aiohttp.ClientSession`` is created in ``setup_hook`` (called by
      discord.py before the gateway connection is opened) and closed in
      ``close``.  It is never recreated — discord.py's built-in reconnect
      logic handles all gateway drops without touching the HTTP session.
    * Rate-limit responses from Discord (HTTP 429) are caught and retried
      after the ``retry_after`` delay reported in the exception.
    * Hugging Face 503 "model loading" responses are retried with backoff
      rather than surfacing an error immediately.
    * All event-handler exceptions are caught so the bot never crashes on a
      single bad message.
    """

    def __init__(self) -> None:
        intents = discord.Intents.default()
        intents.message_content = True
        # connector_owner=False: we manage the connector's lifetime ourselves
        # in setup_hook / close so discord.py doesn't close it prematurely.
        self._connector = aiohttp.TCPConnector()
        super().__init__(
            intents=intents,
            connector=self._connector,
            connector_owner=False,
        )
        self._session: aiohttp.ClientSession | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def setup_hook(self) -> None:
        """Called by discord.py once, before the gateway connection opens.

        This is the correct place to create async resources that must exist
        for the lifetime of the bot.
        """
        self._session = aiohttp.ClientSession(
            connector=self._connector,
            connector_owner=False,
        )
        logger.info("HTTP session created.")

    async def close(self) -> None:
        """Cleanly shut down the HTTP session and connector, then the gateway."""
        logger.info("Shutting down — closing HTTP session …")
        if self._session and not self._session.closed:
            await self._session.close()
        if not self._connector.closed:
            await self._connector.close()
        await super().close()
        logger.info("Shutdown complete.")

    # ------------------------------------------------------------------
    # Discord events
    # ------------------------------------------------------------------

    async def on_ready(self) -> None:
        logger.info("Logged in as %s (ID: %s)", self.user, self.user.id)
        logger.info(
            "Using Hugging Face Inference API (distilgpt2)"
        )

    async def on_message(self, message: discord.Message) -> None:
        # Ignore our own messages
        if message.author == self.user:
            return

        # Only respond to @mentions
        if self.user not in message.mentions:
            return

        # Strip the bot mention(s) from the message text
        content = (
            message.content
            .replace(f"<@{self.user.id}>", "")
            .replace(f"<@!{self.user.id}>", "")
            .strip()
        )

        if not content:
            await self._safe_reply(message, "Hey! Ask me anything.")
            return

        logger.info(
            "Mention from %s in channel %s: %r",
            message.author,
            message.channel,
            content[:80],
        )

        async with message.channel.typing():
            try:
                reply = await self._get_hf_response(content)
            except Exception as exc:
                logger.error("Failed to get HF response: %s", exc)
                await self._safe_reply(
                    message, f"⚠️ {exc}" if isinstance(exc, RuntimeError) else
                    "⚠️ Something went wrong. Please try again later."
                )
                return

        if not reply:
            await self._safe_reply(
                message, "⚠️ Received an empty response. Please try again."
            )
            return

        # Send reply in chunks if it exceeds Discord's 2000-char limit
        chunks = split_message(reply)
        for i, chunk in enumerate(chunks):
            if i == 0:
                await self._safe_reply(message, chunk)
            else:
                await self._safe_send(message.channel, chunk)

    async def on_error(self, event_method: str, *args, **kwargs) -> None:
        """Catch-all so unhandled exceptions in events never crash the bot."""
        logger.exception("Unhandled exception in event '%s'", event_method)

    # ------------------------------------------------------------------
    # Hugging Face API
    # ------------------------------------------------------------------

    async def _get_hf_response(self, user_message: str) -> str:
        """Call the Hugging Face Inference API and return the generated text.

        Retries automatically when the model is still loading (HTTP 503).
        """
        if self._session is None or self._session.closed:
            raise RuntimeError("HTTP session is not available.")

        # distilgpt2 is a completion model: prefix the user message with a
        # simple prompt label so the model has context, then strip the prompt
        # from the returned text before sending it to Discord.
        prompt = f"User: {user_message}\nBot:"
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "return_full_text": False,
            },
        }
        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }

        for attempt in range(1, HF_MODEL_LOADING_MAX_RETRIES + 1):
            async with self._session.post(
                HF_API_URL,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 503:
                    # Model is still loading — wait and retry
                    if attempt < HF_MODEL_LOADING_MAX_RETRIES:
                        logger.warning(
                            "HF model loading (503), retry %d/%d in %ds …",
                            attempt,
                            HF_MODEL_LOADING_MAX_RETRIES,
                            HF_MODEL_LOADING_RETRY_DELAY,
                        )
                        await asyncio.sleep(HF_MODEL_LOADING_RETRY_DELAY)
                        continue
                    raise RuntimeError(
                        "The AI model is still loading after several retries. "
                        "Please try again in a minute."
                    )

                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(
                        f"Hugging Face API returned HTTP {resp.status}: {body[:200]}"
                    )

                data = await resp.json()

                # Response format: [{"generated_text": "..."}]
                if isinstance(data, list) and data:
                    return data[0].get("generated_text", "").strip()

                raise RuntimeError(
                    f"Unexpected HF response format: {str(data)[:200]}"
                )

        # Should never reach here, but satisfy the type checker
        raise RuntimeError("Hugging Face API request failed after all retries.")

    # ------------------------------------------------------------------
    # Helpers — Discord send wrappers that handle rate limits gracefully
    # ------------------------------------------------------------------

    async def _safe_reply(
        self,
        message: discord.Message,
        content: str,
        *,
        max_retries: int = 5,
    ) -> None:
        """Reply to *message*, retrying on Discord rate limits (HTTP 429)."""
        for attempt in range(1, max_retries + 1):
            try:
                await message.reply(content)
                return
            except HTTPException as exc:
                if exc.status == 429:
                    retry_after = getattr(exc, "retry_after", 5.0)
                    logger.warning(
                        "Discord rate limit on reply (attempt %d/%d), "
                        "waiting %.1fs …",
                        attempt,
                        max_retries,
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                else:
                    logger.error("Discord HTTP error on reply: %s", exc)
                    return
        logger.error("Gave up replying after %d rate-limited attempts.", max_retries)

    async def _safe_send(
        self,
        channel: discord.abc.Messageable,
        content: str,
        *,
        max_retries: int = 5,
    ) -> None:
        """Send *content* to *channel*, retrying on Discord rate limits."""
        for attempt in range(1, max_retries + 1):
            try:
                await channel.send(content)
                return
            except HTTPException as exc:
                if exc.status == 429:
                    retry_after = getattr(exc, "retry_after", 5.0)
                    logger.warning(
                        "Discord rate limit on send (attempt %d/%d), "
                        "waiting %.1fs …",
                        attempt,
                        max_retries,
                        retry_after,
                    )
                    await asyncio.sleep(retry_after)
                else:
                    logger.error("Discord HTTP error on send: %s", exc)
                    return
        logger.error("Gave up sending after %d rate-limited attempts.", max_retries)


async def main() -> None:
    """Create the bot and run it, handling fatal startup errors cleanly."""
    bot = DiscordBot()
    try:
        logger.info("Starting Discord bot …")
        await bot.start(DISCORD_TOKEN, reconnect=True)
    except LoginFailure as exc:
        logger.critical("Invalid Discord token — cannot log in: %s", exc)
        raise
    except Exception as exc:
        logger.error("Bot exited with unexpected error: %s", exc)
        raise


asyncio.run(main())


