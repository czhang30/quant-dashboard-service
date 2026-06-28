"""
Position persistence.

Taken trades are stored in positions.json in the project root and survive
app restarts.  Active positions are monitored until the user marks them
closed or they expire naturally.
"""
from __future__ import annotations
import json
import os
import uuid
import datetime as dt

_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")


def load() -> dict:
    if not os.path.exists(_FILE):
        return {"active": [], "closed": []}
    try:
        with open(_FILE) as f:
            return json.load(f)
    except Exception:
        return {"active": [], "closed": []}


def _save(data: dict) -> None:
    with open(_FILE, "w") as f:
        json.dump(data, f, indent=2, default=str)


def add(rec: dict) -> str:
    """Persist a new position and return its ID."""
    data   = load()
    pos_id = str(uuid.uuid4())[:8]
    data["active"].append({
        "id":     pos_id,
        "opened": dt.date.today().isoformat(),
        "status": "open",
        **rec,
    })
    _save(data)
    return pos_id


def close(pos_id: str, reason: str = "manual") -> None:
    data = load()
    for i, pos in enumerate(data["active"]):
        if pos["id"] == pos_id:
            pos.update({
                "status":        "closed",
                "close_reason":  reason,
                "closed":        dt.date.today().isoformat(),
            })
            data["closed"].append(pos)
            data["active"].pop(i)
            break
    _save(data)


def active() -> list[dict]:
    return load()["active"]
