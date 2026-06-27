# Quant watchlist dashboard

A live, self-hosted watchlist dashboard with technical indicators, edge-triggered
alerts, and optional **paper** trading via Alpaca. Built to be extended: each
layer is an independent module you can swap out.

> Not investment advice. This is a decision-support tool. Backtests and live
> signals can mislead. Paper trade for a long time before risking real money —
> and this project intentionally ships with real-money trading disabled.

## Quick start

```bash
cd quant_dashboard
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then edit `config.py` to set your watchlist and alert rules. The dashboard
auto-refreshes every `REFRESH_SECONDS`.

## How it fits together

| File            | Responsibility                                                        |
|-----------------|-----------------------------------------------------------------------|
| `config.py`     | Watchlist, indicator params, alert rules, optional keys. Edit this.   |
| `data.py`       | Fetch OHLCV (yfinance by default). No framework deps.                 |
| `indicators.py` | SMA/EMA, RSI, MACD, ATR, support/resistance, Fibonacci. Pure pandas.  |
| `alerts.py`     | Evaluates rules, edge-triggers, de-duplicates, sends notifications.   |
| `paper.py`      | Optional Alpaca **paper** wrapper. Fails safe if keys/SDK absent.     |
| `app.py`        | Streamlit UI: table, candlestick + levels, alert feed, paper panel.   |

## Alert rules

Rules are declarative dicts in `config.py`. Supported types:

`price_above` · `price_below` · `pct_move_above` · `rsi_above` · `rsi_below` ·
`macd_cross_up` · `macd_cross_down` · `cross_sma_fast_up` · `cross_sma_fast_down` ·
`near_support` · `near_resistance` · `volume_spike` · `near_fib`

Each rule targets `"*"` (all symbols) or a list like `["PATH"]`. Alerts are
**edge-triggered**: a rule fires once when its condition flips from false to
true, then re-arms only after the condition clears — so you don't get spammed
every refresh.

## Notifications

Set `WEBHOOK_URL` in `config.py` to a Discord or Slack incoming webhook to get
pinged outside the dashboard. Console logging is always on.

## Optional: Alpaca paper trading

1. Make a free account at alpaca.markets and grab your **paper** API keys.
2. In `config.py` set `ALPACA_ENABLED = True` and paste the keys.
3. `paper.py` refuses any endpoint without "paper" in the URL, and uses the
   SDK's `paper=True` flag — two independent guards against live trading.
4. To let specific alerts place a simulated order, set `AUTO_PAPER_TRADE = True`
   and list the alert names in `PAPER_TRADE_ON`. Start with `PAPER_ORDER_QTY = 1`.

## Sensible next extensions

- Swap `data.get_history()` to a real-time feed (Alpaca/Polygon) for intraday.
- Add a backtest module that replays `indicators` + `alerts` over history so you
  can measure whether a rule set would have helped before trusting it live.
- Persist the alert feed to SQLite so it survives restarts.
- Add per-rule position sizing using ATR (e.g. risk a fixed % of equity per trade).

## Known limits

- yfinance is delayed/throttled and occasionally returns gaps; fine for swing
  monitoring, not for fast intraday execution.
- The dashboard polls on a timer; it is not an event-driven streaming system.
- `near_fib` / support / resistance use simple rolling windows, not pivot-point
  swing detection — good enough for triage, not precision level-calling.
