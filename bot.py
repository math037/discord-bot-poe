import os
import discord
import fastapi_poe as fp
from dotenv import load_dotenv

load_dotenv()

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
POE_API_KEY = os.environ["POE_API_KEY"]

POE_BOT_NAME = "GPT-4o-Mini"

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


async def get_poe_response(user_message: str) -> str:
    """Send a message to Poe via the fastapi-poe client and return the reply."""
    message = fp.ProtocolMessage(role="user", content=user_message)
    reply_parts: list[str] = []
    async for partial in fp.get_bot_response(
        messages=[message],
        bot_name=POE_BOT_NAME,
        api_key=POE_API_KEY,
    ):
        reply_parts.append(partial.text)
    return "".join(reply_parts)


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
        except fp.BotError as exc:
            print(f"Poe API error: {exc}")
            await message.reply(
                f"⚠️ Poe API returned an error. Please try again later."
            )
        except Exception as exc:
            print(f"Unexpected error: {exc}")
            await message.reply("⚠️ Something went wrong. Please try again later.")


client.run(DISCORD_TOKEN)
