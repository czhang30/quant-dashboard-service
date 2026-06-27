"""
Everything you'd normally tweak lives here. Edit this file, not the engine.

Alert rules are declarative dicts evaluated by alerts.py. Supported `type`s:
    price_above / price_below          -> needs `level`
    pct_move_above                     -> needs `pct`  (abs daily % move)
    rsi_above / rsi_below              -> needs `level`
    macd_cross_up / macd_cross_down    -> no params
    cross_sma_fast_up / _down          -> price crossing the fast SMA
    near_support / near_resistance     -> needs `pct` (within X% of the level)
    volume_spike                       -> needs `ratio` (vol / avg_vol)
    near_fib                           -> needs `pct` and `which` (e.g. "61.8%")
"""

# Tickers you want to watch.
WATCHLIST = ["PATH", "NOW", "MO", "PM", "TSLA", "MSFT", "ORCL", "NOK", "GFS", "SOXL"]

# Data resolution for the dashboard.
DATA_PERIOD = "12mo"      # how much history to pull (yfinance period string)
DATA_INTERVAL = "1d"     # "1d" for swing view; "5m"/"15m" for intraday
REFRESH_SECONDS = 60     # dashboard auto-refresh cadence

# Indicator parameters (passed into indicators.compute_indicators).
INDICATOR_CFG = {
    "rsi_period": 14,
    "ema_fast": 20,
    "sma_fast": 50,
    "sma_slow": 200,
    "sr_lookback": 20,
    "fib_lookback": 15,  # how many bars (data interval, e.g. day) to look back for fib levels
    "vol_lookback": 20,
}

# Alert rules. Each rule can target specific tickers or "*" for all.
ALERT_RULES = [
    {"name": "RSI oversold",      "tickers": "*",        "type": "rsi_below",        "level": 30},
    {"name": "RSI overbought",    "tickers": "*",        "type": "rsi_above",        "level": 70},
    {"name": "MACD turns up",     "tickers": "*",        "type": "macd_cross_up"},
    {"name": "MACD turns down",   "tickers": "*",        "type": "macd_cross_down"},
    {"name": "Big daily move",    "tickers": "*",        "type": "pct_move_above",    "pct": 5.0},
    {"name": "Volume spike",      "tickers": "*",        "type": "volume_spike",      "ratio": 2.0},
    {"name": "Near resistance",   "tickers": "*",        "type": "near_resistance",   "pct": 1.0},
    {"name": "Near support",      "tickers": "*",        "type": "near_support",      "pct": 1.0},
    # Example ticker-specific level alert from our PATH analysis:
    {"name": "PATH reclaims 11.20", "tickers": ["PATH"], "type": "price_above",       "level": 11.20},
    {"name": "PATH at 61.8% fib",   "tickers": ["PATH"], "type": "near_fib", "which": "61.8%", "pct": 0.8},
]

# --- Optional outbound notifications -------------------------------------
# Leave as "" to disable. Webhook works for Discord or Slack incoming hooks.
WEBHOOK_URL = ""

# --- Optional Alpaca PAPER trading ---------------------------------------
# Create a free paper account at alpaca.markets, then paste the PAPER keys.
# These must be the *paper* keys; paper.py refuses to run against live URLs.
ALPACA_ENABLED = False
ALPACA_API_KEY = ""
ALPACA_SECRET_KEY = ""
ALPACA_PAPER_URL = "https://paper-api.alpaca.markets"

# When an alert fires, optionally place a small paper order. Off by default
# so the dashboard is observe-only until you explicitly opt in.
AUTO_PAPER_TRADE = False
PAPER_ORDER_QTY = 1            # shares per simulated order
PAPER_TRADE_ON = ["RSI oversold"]  # which alert names trigger a paper buy
