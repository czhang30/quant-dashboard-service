"""
Alert engine. Evaluates declarative rules from config against the indicator
snapshot, de-duplicates so a rule fires once per condition-crossing (not every
refresh while the condition stays true), and dispatches notifications.

State is a dict you own and persist (e.g. Streamlit session_state):
    state[(ticker, rule_name)] = True/False  # was the condition true last time
"""
from __future__ import annotations

import datetime as dt

import numpy as np

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


def _condition_met(rule: dict, ind: dict) -> bool:
    t = rule["type"]
    price = ind.get("price", np.nan)

    if t == "price_above":
        return price > rule["level"]
    if t == "price_below":
        return price < rule["level"]
    if t == "pct_move_above":
        return abs(ind.get("pct_change", 0) or 0) >= rule["pct"]
    if t == "rsi_above":
        return ind.get("rsi", 50) >= rule["level"]
    if t == "rsi_below":
        return ind.get("rsi", 50) <= rule["level"]
    if t == "macd_cross_up":
        return bool(ind.get("macd_cross_up"))
    if t == "macd_cross_down":
        return bool(ind.get("macd_cross_down"))
    if t == "cross_sma_fast_up":
        return ind.get("above_sma_fast") is True
    if t == "cross_sma_fast_down":
        return ind.get("above_sma_fast") is False
    if t == "near_support":
        d = ind.get("dist_to_support_pct")
        return d is not None and 0 <= d <= rule["pct"]
    if t == "near_resistance":
        d = ind.get("dist_to_resistance_pct")
        return d is not None and -rule["pct"] <= d <= 0
    if t == "volume_spike":
        return (ind.get("vol_ratio") or 0) >= rule["ratio"]
    if t == "near_fib":
        lvl = ind.get("fib", {}).get(rule["which"])
        if lvl is None or not lvl:
            return False
        return abs(price / lvl - 1) * 100 <= rule["pct"]

    # --- quant indicators ------------------------------------------------
    if t == "bb_pct_b_above":
        v = ind.get("bb_pct_b")
        return v is not None and not np.isnan(v) and v >= rule["level"]
    if t == "bb_pct_b_below":
        v = ind.get("bb_pct_b")
        return v is not None and not np.isnan(v) and v <= rule["level"]
    if t == "zscore_above":
        v = ind.get("zscore")
        return v is not None and not np.isnan(v) and v >= rule["level"]
    if t == "zscore_below":
        v = ind.get("zscore")
        return v is not None and not np.isnan(v) and v <= rule["level"]
    if t == "adx_above":
        v = ind.get("adx")
        return v is not None and not np.isnan(v) and v >= rule["level"]
    if t == "adx_below":
        v = ind.get("adx")
        return v is not None and not np.isnan(v) and v < rule["level"]
    if t == "mom_rank_top":
        rank = ind.get("mom_rank")
        return rank is not None and rank <= rule["n"]
    if t == "mom_rank_bottom":
        rank = ind.get("mom_rank")
        total = ind.get("mom_rank_total", 0)
        return rank is not None and rank >= total - rule["n"] + 1

    return False


def _rule_targets(rule: dict, ticker: str) -> bool:
    tg = rule.get("tickers", "*")
    return tg == "*" or ticker in tg


def evaluate(ticker: str, ind: dict, rules: list[dict], state: dict) -> list[dict]:
    """
    Return newly-triggered alerts (edge-triggered: only when a condition flips
    from false to true). Mutates `state` in place.
    """
    fired = []
    for rule in rules:
        if not _rule_targets(rule, ticker):
            continue
        key = (ticker, rule["name"])
        now = _condition_met(rule, ind)
        was = state.get(key, False)
        if now and not was:
            fired.append(
                {
                    "ticker": ticker,
                    "name": rule["name"],
                    "type": rule["type"],
                    "price": ind.get("price"),
                    "time": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
            )
        state[key] = now
    return fired


def notify(alert: dict, webhook_url: str = "") -> None:
    """Console always; webhook (Discord/Slack incoming hook) if configured."""
    msg = f"[{alert['time']}] {alert['ticker']}: {alert['name']} @ ${alert['price']:.2f}"
    print("ALERT:", msg)
    if webhook_url and requests is not None:
        try:
            requests.post(webhook_url, json={"text": msg, "content": msg}, timeout=5)
        except Exception as e:  # pragma: no cover
            print("webhook failed:", e)
