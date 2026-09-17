# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yevhen Potapov. All rights reserved.
# Part of ContentForge. See LICENSE in the repository root.
"""desc_block.py -- one place that builds the YouTube description footer:
channel + code links + an honest "how this video was made" note.

    from desc_block import build_description
    desc = build_description(plan_description, lang="ru")
"""
from __future__ import annotations

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


def links_block(lang: str = "ru") -> str:
    label = "Код" if lang == "ru" else "Code"
    return "\n".join([
        f"{CHANNEL_NAME} {CHANNEL_HANDLE}",
        CHANNEL_URL,
        "",
        f"GitHub: {GITHUB_URL}",
        f"{label} / Gold Standard: {GOLD_URL}",
        f"ContentForge (multi-platform factory): {CONTENTFORGE_URL}",
    ])


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
