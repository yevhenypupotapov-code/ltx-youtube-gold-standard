#!/usr/bin/env python3
# Copyright (c) 2026 Yevhen Potapov. All rights reserved.
# Part of ContentForge. See LICENSE in the repository root.
"""Add Edge-TTS voiceover to a silent LTX mp4.

Never loops the video under a longer VO (that reads as AI spam).
If audio is longer: pad with a slow Ken Burns hold on the last frame.
If video is longer: trim with -shortest.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

SYS_PY = Path(r"C:\Users\yevhe\AppData\Local\Python\pythoncore-3.14-64\python.exe")
FFMPEG = Path(r"C:\Users\yevhe\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe")
VOICES = {"ru": "ru-RU-DmitryNeural", "en": "en-US-GuyNeural"}
# Keep VO roughly matched to short LTX montages (~45–90s speech)
MAX_CHARS = 1400


def _ffmpeg() -> str:
    if FFMPEG.exists():
        return str(FFMPEG)
    return "ffmpeg"


def probe_duration(path: Path) -> float:
    cmd = [
        _ffmpeg().replace("ffmpeg", "ffprobe") if False else "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        str(path),
    ]
    # Prefer sibling ffprobe next to ffmpeg
    ff = Path(_ffmpeg())
    probe = ff.with_name("ffprobe.exe" if ff.suffix.lower() == ".exe" else "ffprobe")
    if probe.exists():
        cmd[0] = str(probe)
    else:
        cmd[0] = "ffprobe"
    raw = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    data = json.loads(raw)
    return float(data.get("format", {}).get("duration") or 0.0)



def _collapse_vo_loops(text: str) -> str:
    """Last-chance VO sanitizer: drop duplicated closing phrases before TTS."""
    import re as _re
    words = (text or "").split()
    n = len(words)
    if n >= 16:
        for k in range(n // 2, 7, -1):
            a = " ".join(words[-k:]).lower()
            b = " ".join(words[-2 * k:-k]).lower()
            if a == b:
                words = words[:-k]
                break
    text = " ".join(words)
    # collapse exact repeated sentences
    parts = [p.strip() for p in _re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    out = []
    seen = set()
    for p in parts:
        key = _re.sub(r"\W+", " ", p.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return " ".join(out) if out else text


WORKSPACE = Path(r"C:\Users\yevhe\.openclaw\workspace")


def tts_to_mp3(text: str, dest: Path, lang: str) -> Path:
    text = _collapse_vo_loops(text)
    dest.parent.mkdir(parents=True, exist_ok=True)
    text = " ".join((text or "").split())[:MAX_CHARS]
    if not text:
        raise ValueError("empty TTS text")

    # Local-first: system voice, no internet. LTX_TTS_MODE=cloud restores edge-tts.
    if os.environ.get("LTX_TTS_MODE", "local").lower() != "cloud":
        try:
            if str(WORKSPACE) not in sys.path:
                sys.path.insert(0, str(WORKSPACE))
            from local_tts import synth as _local_synth
            if lang == "ru":
                try:
                    from ru_speech import speakable_ru as _sru
                    _b = text
                    text = _sru(text)
                    if text != _b:
                        print("[audio] ru speech normalised (latin -> russian)", flush=True)
                except Exception as _sr:
                    print(f"[audio] ru_speech skipped ({_sr})", flush=True)
            _local_synth(text, dest, lang=lang)
            return dest
        except Exception as exc:
            print(f"[audio] local TTS failed ({exc}) -> cloud fallback", flush=True)

    voice = VOICES.get(lang, VOICES["en"])
    py = str(SYS_PY if SYS_PY.exists() else sys.executable)
    helper = dest.with_suffix(".tts.py")
    helper.write_text(
        "import asyncio\n"
        "import edge_tts\n"
        f"TEXT = {text!r}\n"
        f"VOICE = {voice!r}\n"
        f"OUT = {str(dest)!r}\n"
        "asyncio.run(edge_tts.Communicate(TEXT, VOICE).save(OUT))\n",
        encoding="utf-8",
    )
    try:
        subprocess.check_call([py, str(helper)])
    finally:
        helper.unlink(missing_ok=True)
    if not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(f"TTS failed: {dest}")
    return dest


def _pad_kenburns(video: Path, target_secs: float, dest: Path) -> Path:
    """Extend video to target_secs by holding last frame with a slow zoom (no clip loop)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    vd = max(0.1, probe_duration(video))
    extra = max(0.0, target_secs - vd)
    if extra < 0.15:
        if dest.resolve() != video.resolve():
            dest.write_bytes(video.read_bytes())
        return dest
    # freeze last frame, then slow zoom over full length
    # tpad stop_mode=clone adds frozen frames; zoompan does Ken Burns on the pad stretch
    filt = (
        f"tpad=stop_mode=clone:stop_duration={extra:.3f},"
        f"scale=1280:720:force_original_aspect_ratio=decrease,"
        f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
        f"zoompan=z='min(1.08,1+0.0008*on)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1280x720:fps=24"
    )
    cmd = [
        _ffmpeg(), "-y", "-i", str(video),
        "-vf", filt,
        "-t", f"{target_secs:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart", str(dest),
    ]
    subprocess.check_call(cmd)
    return dest


def mux_fit(video: Path, audio: Path, dest: Path) -> Path:
    """Mux without looping the source montage."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    vd = probe_duration(video)
    ad = probe_duration(audio)
    print(f"[audio] durations video={vd:.2f}s audio={ad:.2f}s", flush=True)
    work = video
    padded = video.with_name(video.stem + "_pad.mp4")
    if ad > vd + 0.25:
        pad_extra = ad - vd
        # Hard rule: last-frame pad must NOT create an effective still hold > MAX_STILL_HOLD_SEC.
        # Assembler is responsible for covering audio with <=5s clips; only tiny A/V slop allowed.
        MAX_PAD = 0.35
        if pad_extra > MAX_PAD:
            print(
                f"[audio] CRITICAL: pad would hold last frame +{pad_extra:.1f}s > MAX_PAD={MAX_PAD}s "
                f"(would stack on last clip and break max_still_hold_sec=5.0); "
                "refusing stub stretch (assembler should have covered)",
                flush=True,
            )
            raise RuntimeError(
                f"audio pad {pad_extra:.1f}s exceeds MAX_PAD={MAX_PAD} — quality abort"
            )
        print(f"[audio] pad last-frame Ken Burns +{pad_extra:.1f}s (no clip loop)", flush=True)
        work = _pad_kenburns(video, ad, padded)
    cmd = [
        _ffmpeg(), "-y",
        "-i", str(work),
        "-i", str(audio),
        "-shortest",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-r", "24",
        "-c:a", "aac",
        "-b:a", "160k",
        "-ar", "44100",
        "-ac", "2",
        "-movflags", "+faststart",
        str(dest),
    ]
    subprocess.check_call(cmd)
    if not dest.exists() or dest.stat().st_size < 1000:
        raise RuntimeError(f"mux failed: {dest}")
    return dest


def fit_script_to_seconds(text: str, seconds: float, lang: str = "en") -> str:
    """Rough trim so VO does not dwarf the montage."""
    text = " ".join((text or "").split())
    if not text or seconds <= 0:
        return text
    wps = 2.2 if lang == "ru" else 2.5  # words per second
    budget = max(40, int(seconds * wps))
    words = text.split()
    if len(words) <= budget:
        return text
    cut = " ".join(words[:budget]).rstrip(" ,;")
    if not cut.endswith("."):
        cut += "."
    print(f"[audio] trimmed script {len(words)}->{budget} words for ~{seconds:.0f}s video", flush=True)
    return cut


def add_voiceover(video: Path, text: str, lang: str = "en", *, fit_to_video: bool = True) -> Path:
    video = Path(video)
    text = (text or "").strip()
    if not text:
        print("[audio] no script, leaving silent", flush=True)
        return video
    if fit_to_video:
        try:
            vd = probe_duration(video)
            # Allow slight overage; Ken Burns pads the rest
            text = fit_script_to_seconds(text, max(vd * 1.15, vd + 2.0), lang)
        except Exception as exc:
            print(f"[audio] duration probe skipped ({exc})", flush=True)
    tts = video.with_name(video.stem + "_tts.mp3")
    out = video.with_name(video.stem + "_vo.mp4")
    text = _collapse_vo_loops(text)
    print(f"[audio] TTS {lang} chars={len(text)}", flush=True)
    tts_to_mp3(text, tts, lang)
    print(f"[audio] mux_fit {video.name} + {tts.name}", flush=True)
    mux_fit(video, tts, out)
    print(f"[audio] {out}", flush=True)
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("video")
    p.add_argument("--text", required=True)
    p.add_argument("--lang", default="en")
    args = p.parse_args()
    print(add_voiceover(Path(args.video), args.text, args.lang))
