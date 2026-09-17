# -*- coding: utf-8 -*-
"""visual_identity.py -- give every release a distinct look.

The factory does not repeat within a video; it repeats BETWEEN videos: same
dark desk, same light, same grade. This picks one colour grade per release,
never reusing the last N releases, and records the choice in pipeline_state.json.

    from visual_identity import current_identity
    name, flt = current_identity()      # ('cool-night', 'colorbalance=...')
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
# Separate file: pipeline_state.json is rewritten by the assembler and would
# otherwise clobber the identity history between runs.
STATE = ROOT / "visual_identity_history.json"
HISTORY_KEY = "visual_identity_history"
AVOID_LAST = int(os.environ.get("LTX_VISUAL_AVOID_LAST", "6"))

# name -> ffmpeg filter chain ("" = untouched). Kept subtle so the footage
# still reads as the same channel, but clearly different between releases.
GRADES: list[tuple[str, str]] = [
    # "neutral" last: the first release should already look graded, not flat.
    ("cool-night", "colorbalance=rs=-0.07:gs=-0.02:bs=0.08,eq=contrast=1.05:saturation=1.02"),
    ("warm-amber", "colorbalance=rs=0.08:gs=0.02:bs=-0.07,eq=contrast=1.03:saturation=1.06"),
    ("teal-shadow", "colorbalance=rs=-0.03:gs=0.03:bs=0.06,eq=contrast=1.07"),
    ("dusk-magenta", "colorbalance=rs=0.06:gs=-0.03:bs=0.07,eq=contrast=1.05"),
    ("steel-mono", "hue=s=0.25,eq=contrast=1.12:brightness=0.01"),
    ("emerald-dim", "colorbalance=rs=-0.05:gs=0.06:bs=-0.02,eq=contrast=1.04:saturation=1.03"),
    ("neutral", ""),
]
_GRADE_MAP = dict(GRADES)
_CACHE: dict = {}


def _load() -> dict:
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(state: dict) -> None:
    try:
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def _pick(state: dict) -> tuple[str, str]:
    hist = list(state.get(HISTORY_KEY) or [])
    recent = [str(h.get("name")) for h in hist[-AVOID_LAST:] if isinstance(h, dict)]
    names = [g[0] for g in GRADES]
    counts = {n: 0 for n in names}
    for h in hist:
        n = h.get("name") if isinstance(h, dict) else None
        if n in counts:
            counts[n] += 1
    order = sorted(names, key=lambda n: (counts[n], names.index(n)))
    for n in order:
        if n not in recent:
            return n, _GRADE_MAP[n]
    n = order[0]
    return n, _GRADE_MAP[n]


def current_identity(record: bool = True) -> tuple[str, str]:
    """Pick (and by default record) this run's grade. Cached per process."""
    if "ident" in _CACHE:
        return _CACHE["ident"]
    state = _load()
    name, flt = _pick(state)
    if record:
        hist = list(state.get(HISTORY_KEY) or [])
        hist.append({"name": name, "ts": datetime.now().isoformat(timespec="seconds")})
        state[HISTORY_KEY] = hist[-40:]
        _save(state)
    _CACHE["ident"] = (name, flt)
    return _CACHE["ident"]


if __name__ == "__main__":
    n, f = current_identity(record=False)
    print(f"next identity: {n}\nfilter: {f or '(none)'}")
    st = _load()
    print("history:", [h.get("name") for h in (st.get(HISTORY_KEY) or [])][-10:])
