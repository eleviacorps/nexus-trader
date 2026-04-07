import asyncio
import unittest

from src.v18.websocket_feed import LiveFeedManager, seconds_to_next_15m


class _FakeWebSocket:
    def __init__(self) -> None:
        self.accepted = False
        self.sent = []

    async def accept(self) -> None:
        self.accepted = True

    async def send_json(self, payload) -> None:
        self.sent.append(payload)


class V18WebSocketTests(unittest.IsolatedAsyncioTestCase):
    async def test_feed_manager_connects_broadcasts_and_disconnects(self) -> None:
        manager = LiveFeedManager()
        ws = _FakeWebSocket()
        await manager.connect(ws, symbol="XAUUSD")
        self.assertTrue(ws.accepted)
        await manager.broadcast({"ok": True})
        self.assertEqual(ws.sent[-1]["ok"], True)
        manager.disconnect(ws)
        self.assertEqual(len(manager._connections), 0)

    async def test_heartbeat_loop_emits_tick_payload(self) -> None:
        manager = LiveFeedManager()
        ws = _FakeWebSocket()
        await manager.connect(ws, symbol="XAUUSD")
        task = asyncio.create_task(manager.heartbeat_loop(price_fn=lambda _symbol: 2350.0, sqt_fn=lambda: {"label": "HOT"}))
        await asyncio.sleep(1.1)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        self.assertTrue(ws.sent)
        self.assertEqual(ws.sent[-1]["type"], "tick")
        self.assertEqual(ws.sent[-1]["symbol"], "XAUUSD")
        self.assertIn("paper_summary", ws.sent[-1])

    def test_seconds_to_next_15m_is_bounded(self) -> None:
        remaining = seconds_to_next_15m()
        self.assertGreaterEqual(remaining, 1)
        self.assertLessEqual(remaining, 900)


if __name__ == "__main__":
    unittest.main()
