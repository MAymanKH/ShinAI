import asyncio
import sys


async def run() -> None:
    # Imported inside the coroutine: Pyrogram captures the running event loop
    # when its Client is constructed, so the application must be built after
    # asyncio.run() has created the loop that will also shut it down.
    from shin_ai.app import Application
    from shin_ai.services.social import index_social_context
    from shin_ai.utils.logger_config import logger

    application = Application.build()

    try:
        await index_social_context()
    except Exception as error:
        logger.error("Failed to index social context: %s", error)

    await application.start()

    from shin_ai.core.lifecycle import wait_for_shutdown

    try:
        await wait_for_shutdown()
    finally:
        await application.shutdown()


def main() -> int:
    from shin_ai.settings import load_settings
    from shin_ai.utils.logger_config import configure_logging

    try:
        settings = load_settings()
    except (FileNotFoundError, ValueError) as error:
        # Nothing is configured yet, so report plainly on stderr rather than
        # through a logger whose own configuration just failed to load.
        print(f"Configuration error: {error}", file=sys.stderr)
        return 1

    configure_logging(settings.logging)

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
