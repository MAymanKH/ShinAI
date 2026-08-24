import asyncio
import signal

from shin_ai.core.lifecycle import wait_for_shutdown


def test_first_signal_requests_shutdown_and_restores_signal_handlers(monkeypatch) -> None:
    async def scenario() -> None:
        loop = asyncio.get_running_loop()
        handlers = {}
        removed = []

        def add_signal_handler(received_signal, callback, *args) -> None:
            handlers[received_signal] = (callback, args)

        def remove_signal_handler(received_signal) -> bool:
            removed.append(received_signal)
            handlers.pop(received_signal, None)
            return True

        monkeypatch.setattr(loop, "add_signal_handler", add_signal_handler)
        monkeypatch.setattr(loop, "remove_signal_handler", remove_signal_handler)

        waiter = asyncio.create_task(wait_for_shutdown())
        await asyncio.sleep(0)

        callback, args = handlers[signal.SIGINT]
        callback(*args)
        await waiter

        assert set(removed) == {signal.SIGINT, signal.SIGTERM}
        assert handlers == {}

    asyncio.run(scenario())
