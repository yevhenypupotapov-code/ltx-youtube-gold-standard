# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yevhen Potapov. All rights reserved.
# Part of ContentForge. See LICENSE in the repository root.
"""desc_block.py -- one place that builds the YouTube description footer:
channel + code links + an honest "how this video was made" note.

    from desc_block import build_description
    desc = build_description(plan_description, lang="ru")
"""
from __future__ import annotations

import json
import os
from pathlib import Path

CHANNEL_NAME = "YEVHEN POTAPOV"
CHANNEL_HANDLE = "@yevhenpotapov5956"
CHANNEL_URL = "https://www.youtube.com/@yevhenpotapov5956"
GITHUB_URL = "https://github.com/yevhenypupotapov-code"
GOLD_URL = "https://github.com/yevhenypupotapov-code/ltx-youtube-gold-standard"
CONTENTFORGE_URL = "https://github.com/yevhenypupotapov-code/contentforge"

HOW_RU = (
    "Как сделано\n"
    "Видео собрано полностью на локальном ПК, без облачных генераторов:\n"
    "• сценарий и раскадровка — локальная языковая модель (Ollama) на этом же ПК\n"
    "• кадры и анимация — ComfyUI + LTXV\n"
    "• озвучка — системный синтез речи Windows (SAPI), офлайн\n"
    "• монтаж и титры — FFmpeg\n"
    "• тема выпуска — актуальные тренды из открытых лент (Hacker News, Google News, Reddit)\n"
    "Публикация — YouTube Data API."
)

HOW_EN = (
    "How it was made\n"
    "Built entirely on a local PC, no cloud generators:\n"
    "- script and shot list: local LLM (Ollama) on this machine\n"
    "- visuals and motion: ComfyUI + LTXV\n"
    "- voiceover: Windows system speech (SAPI), offline\n"
    "- editing and captions: FFmpeg\n"
    "- topic: live trends from open feeds (Hacker News, Google News, Reddit)\n"
    "Upload: YouTube Data API."
)



# ── site link ───────────────────────────────────────────────────────────────
PROJECTS_JSON = Path(r"C:\Users\yevhe\.openclaw-autoclaw\workspace\projects\projects.json")


def site_url() -> str:
    """Public address of the project site.

    Manual override first, then whatever AutoClaw wrote into projects.json:
    the stable address once the site is published, the preview one before that.
    Returns "" when nothing is known yet — the line is then simply omitted.
    """
    env = os.environ.get("CONTENTFORGE_SITE_URL", "").strip()
    if env:
        return env
    try:
        import json as _json
        data = _json.loads(PROJECTS_JSON.read_text(encoding="utf-8"))
    except Exception:
        return ""
    if not isinstance(data, list):
        return ""
    for proj in data:
        res = (proj or {}).get("deploymentResult") or {}
        if not res:
            continue
        status = str(res.get("status") or "").lower()
        stable = str(res.get("stableUrl") or "").strip()
        preview = str(res.get("previewUrl") or "").strip()
        if ("publish" in status or "stable" in status) and stable:
            return stable
        if preview:
            return preview
        if stable:
            return stable
    return ""

def links_block(lang: str = "ru") -> str:
    label = "Код" if lang == "ru" else "Code"
    site_label = "Сайт" if lang == "ru" else "Website"
    rows = []
    url = site_url()
    if url:
        rows += [f"{site_label}: {url}", ""]
    rows += [
        f"{CHANNEL_NAME} {CHANNEL_HANDLE}",
        CHANNEL_URL,
        "",
        f"GitHub: {GITHUB_URL}",
        f"{label} / Gold Standard: {GOLD_URL}",
        f"ContentForge (multi-platform factory): {CONTENTFORGE_URL}",
    ]
    return "\n".join(rows)


def build_description(body: str = "", lang: str = "ru", extra: str = "") -> str:
    how = HOW_RU if lang == "ru" else HOW_EN
    chunks = []
    base = (body or "").strip()
    if base:
        chunks.append(base)
    if extra.strip():
        chunks.append(extra.strip())
    chunks.append(how)
    chunks.append(links_block(lang))
    return "\n\n".join(chunks)[:5000]


if __name__ == "__main__":
    print(build_description("Короткое описание выпуска про ИИ.", lang="ru"))
