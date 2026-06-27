"""
Thin, optional wrapper around Alpaca's PAPER trading API.

Design goals:
  * Fails safe. If the SDK is missing or keys are blank, every method is a
    no-op and is_enabled() returns False, so the dashboard still runs.
  * Refuses to point at anything that isn't a paper endpoint, as a guard
    against accidentally wiring live keys into an automated loop.
"""
from __future__ import annotations

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
    _SDK = True
except ImportError:  # pragma: no cover
    _SDK = False


class PaperBroker:
    def __init__(self, api_key: str, secret_key: str, paper_url: str, enabled: bool):
        self._client = None
        self._enabled = False
        if not (enabled and _SDK and api_key and secret_key):
            return
        if "paper" not in paper_url:
            raise ValueError("Refusing to run: endpoint is not a paper URL.")
        # paper=True is the real safety switch; the URL check is belt-and-braces.
        self._client = TradingClient(api_key, secret_key, paper=True)
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def account(self) -> dict:
        if not self._enabled:
            return {}
        a = self._client.get_account()
        return {
            "equity": float(a.equity),
            "cash": float(a.cash),
            "buying_power": float(a.buying_power),
            "status": str(a.status),
        }

    def positions(self) -> list[dict]:
        if not self._enabled:
            return []
        return [
            {
                "symbol": p.symbol,
                "qty": float(p.qty),
                "avg_price": float(p.avg_entry_price),
                "market_value": float(p.market_value),
                "unrealized_pl": float(p.unrealized_pl),
                "unrealized_plpc": float(p.unrealized_plpc) * 100,
            }
            for p in self._client.get_all_positions()
        ]

    def market_buy(self, symbol: str, qty: int = 1) -> dict:
        if not self._enabled:
            return {"ok": False, "reason": "paper trading disabled"}
        try:
            order = self._client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            return {"ok": True, "id": str(order.id), "symbol": symbol, "qty": qty}
        except Exception as e:  # pragma: no cover
            return {"ok": False, "reason": str(e)}
