"""Manual trade modal screen."""

from __future__ import annotations

from typing import Any

from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Static

from nexus_packaged.trading.manual_trader import ManualOrderRequest


class ManualTradeModal(ModalScreen):
    """Manual trade modal with live validation."""

    CSS = """
    ManualTradeModal {
        align: center middle;
    }
    #manual-root {
        width: 94%;
        height: 94%;
        border: round $accent;
        padding: 1 2;
        background: $surface;
    }
    .row {
        height: auto;
        margin: 0 0 1 0;
    }
    .row Label {
        width: 18;
    }
    .row Input {
        width: 1fr;
    }
    #open-positions {
        height: 1fr;
        border: solid $accent;
        padding: 1;
        overflow-y: auto;
    }
    """

    def __init__(self, manual_trader, inference_runner, settings: dict, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.manual_trader = manual_trader
        self.inference_runner = inference_runner
        self.settings = settings
        self._message = Static("", id="manual-message")
        self._open_positions = Static("(none)", id="open-positions")

    def compose(self):
        auto_cfg = dict(self.settings.get("auto_trade", {}))
        default_lot = float(auto_cfg.get("fixed_lot_size", 0.01))
        with VerticalScroll(id="manual-root"):
            yield Label("MANUAL TRADE")
            yield Static("Symbol: XAUUSD", id="symbol")
            yield Static("Price: --", id="price")

            with Horizontal(classes="row"):
                yield Label("Direction")
                yield Input("BUY", id="direction")
            with Horizontal(classes="row"):
                yield Label("Order Type")
                yield Input("market", id="entry_type")
            with Horizontal(classes="row"):
                yield Label("Entry Price")
                yield Input("", placeholder="blank = market", id="entry_price")
            with Horizontal(classes="row"):
                yield Label("Lot Size")
                yield Input(f"{default_lot:.2f}", id="lot_size")
            with Horizontal(classes="row"):
                yield Label("Stop Loss")
                yield Input("", id="sl")
            with Horizontal(classes="row"):
                yield Label("Take Profit")
                yield Input("", id="tp")
            with Horizontal(classes="row"):
                yield Label("Comment")
                yield Input("", id="comment")

            yield Static("Risk/Reward: --  Est. Risk: --  Est. Margin: --", id="risk-line")
            yield Static("Model Signal: --  Confidence: --", id="model-line")
            yield self._message

            with Horizontal(classes="row"):
                yield Button("PLACE ORDER", id="place", variant="primary")
                yield Button("CANCEL", id="cancel")

            yield Label("Open Positions")
            yield self._open_positions

    def on_mount(self) -> None:
        self.set_interval(1.0, self._refresh_live)

    def _refresh_live(self) -> None:
        price = float(self.inference_runner.current_price())
        self.query_one("#price", Static).update(f"Price: {price:.2f}")
        event = self.inference_runner.latest_event
        if event:
            self.query_one("#model-line", Static).update(
                f"Model Signal: {event.signal}  Confidence: {event.confidence:.2f}"
            )
        open_trades = self.manual_trader.trade_manager.get_open_trades()
        if open_trades:
            lines = []
            for trade in open_trades[-20:]:
                lines.append(
                    f"{trade.trade_id[:8]} {trade.direction} lot={trade.lot_size:.2f} "
                    f"entry={trade.entry_price:.2f} sl={trade.sl:.2f} tp={trade.tp:.2f}"
                )
            self._open_positions.update("\n".join(lines))
        else:
            self._open_positions.update("(none)")
        self._update_estimates(price)

    def _update_estimates(self, price: float) -> None:
        try:
            lot = float(self.query_one("#lot_size", Input).value or "0")
            entry_value = self.query_one("#entry_price", Input).value.strip()
            entry = float(entry_value) if entry_value else price
            sl_text = self.query_one("#sl", Input).value.strip()
            sl = float(sl_text) if sl_text else 0.0
            rr = float(self.settings.get("auto_trade", {}).get("risk_reward", 2.0))
            direction = self.query_one("#direction", Input).value.strip().upper() or "BUY"
            tp_input = self.query_one("#tp", Input)

            # Populate TP from RR if TP is blank and SL is provided.
            if sl > 0 and not tp_input.value.strip():
                sl_distance = abs(entry - sl)
                if direction == "BUY":
                    tp_input.value = f"{entry + sl_distance * rr:.2f}"
                else:
                    tp_input.value = f"{entry - sl_distance * rr:.2f}"

            pip_value = float(self.settings.get("broker", {}).get("pip_value_per_lot", 1.0))
            leverage = float(self.settings.get("backtest", {}).get("default_leverage", 200))
            contract_size = float(self.settings.get("broker", {}).get("contract_size", 100.0))
            sl_pips = abs(entry - sl) / 0.01 if sl > 0 else 0.0
            est_risk = lot * sl_pips * pip_value
            est_margin = (lot * contract_size * max(entry, 1e-9)) / max(1.0, leverage)
            self.query_one("#risk-line", Static).update(
                f"Risk/Reward: {rr:.2f}  Est. Risk: ${est_risk:.2f}  Est. Margin: ${est_margin:.2f}"
            )
        except Exception:
            self.query_one("#risk-line", Static).update("Risk/Reward: --  Est. Risk: --  Est. Margin: --")

    def _build_request(self) -> ManualOrderRequest:
        direction = self.query_one("#direction", Input).value.strip().upper() or "BUY"
        entry_type = self.query_one("#entry_type", Input).value.strip().lower() or "market"
        lot_size = float(self.query_one("#lot_size", Input).value or "0.01")
        entry_raw = self.query_one("#entry_price", Input).value.strip()
        entry_price = float(entry_raw) if entry_raw else None
        sl = float(self.query_one("#sl", Input).value or "0")
        tp = float(self.query_one("#tp", Input).value or "0")
        comment = self.query_one("#comment", Input).value.strip()
        return ManualOrderRequest(
            direction=direction,
            lot_size=lot_size,
            entry_type=entry_type,
            entry_price=entry_price,
            sl=sl,
            tp=tp,
            comment=comment,
            expiry=None,
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "cancel":
            self.dismiss(None)
            return
        if event.button.id == "place":
            try:
                request = self._build_request()
                self.manual_trader.place_trade(request)
                self._message.update("Order placed successfully.")
                self.dismiss(True)
            except Exception as exc:  # noqa: BLE001
                self._message.update(f"Error: {exc}")
