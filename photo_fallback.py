# -*- coding: utf-8 -*-
"""Photoreal desk/keyboard stills from photo_pool — no abstract glowing rectangles."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance


def make_photoreal_still(
    dest: Path,
    *,
    beat: str,
    title: str,
    index: int,
    width: int = 1280,
    height: int = 720,
) -> Path:
    from shot_gate import is_geometric_still, still_luma_ok, passes_ocr_gate

    dest.parent.mkdir(parents=True, exist_ok=True)
    seed = int(hashlib.md5(f"{index}:{beat}:{title}".encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    root = Path(__file__).resolve().parent
    photo_dirs = [root / "fallback_pool", root / "photo_pool", root / "fallback_pool" / "photos"]
    photos: list[Path] = []
    for d in photo_dirs:
        if d.is_dir():
            photos.extend(sorted(d.glob("*.jpg")))
            photos.extend(sorted(d.glob("*.jpeg")))
            photos.extend(sorted(d.glob("*.png")))
    photos = [
        p for p in photos
        if p.is_file() and p.stat().st_size > 20_000
        and "_purged" not in str(p).lower()
        and ".bak." not in p.name.lower()
    ]
    # Prefer fallback_pool root assets (preflight-gated) first
    photos.sort(key=lambda p: (0 if p.parent.name == "fallback_pool" else 1, p.name))
    if not photos:
        raise RuntimeError(
            "make_fallback_still: photo_pool empty — refuse abstract rect stubs. "
            "Populate ltx-youtube/photo_pool with desk/keyboard photos."
        )

    def _render(src: Path) -> Path:
        img = Image.open(src).convert("RGB")
        W0, H0 = img.size
        aspect = width / float(height)
        zoom = float(1.0 + (seed % 25) * 0.01)
        crop_w = min(W0, int(W0 / zoom))
        crop_h = min(H0, int(crop_w / aspect))
        if crop_h > H0:
            crop_h = H0
            crop_w = int(crop_h * aspect)
        max_x = max(0, W0 - crop_w)
        max_y = max(0, H0 - crop_h)
        x0 = int(rng.integers(0, max_x + 1)) if max_x else 0
        y0 = int(rng.integers(0, max_y + 1)) if max_y else 0
        img = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        bright = 0.95 + ((seed % 7) * 0.02)
        contrast = 0.98 + ((seed % 5) * 0.015)
        color = 0.92 + ((seed % 6) * 0.02)
        img = ImageEnhance.Brightness(img).enhance(bright)
        img = ImageEnhance.Contrast(img).enhance(contrast)
        img = ImageEnhance.Color(img).enhance(color)
        arr = np.asarray(img).astype(np.float32)
        arr = np.clip(arr + rng.normal(0, 2.5, arr.shape), 0, 255)
        img = Image.fromarray(arr.astype(np.uint8))
        tmp = dest.with_suffix(".tmp.jpg")
        img.save(tmp, quality=92)
        return tmp

    src = photos[(index - 1 + seed) % len(photos)]
    tmp = _render(src)
    geo, gsc = is_geometric_still(tmp)
    if geo:
        tmp.unlink(missing_ok=True)
        src2 = photos[(index + 3 + seed) % len(photos)]
        tmp = _render(src2)
        geo, gsc = is_geometric_still(tmp)
        if geo:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"make_fallback_still: photo still geometric glow_d={gsc.get('glow_delta', 0):.1f} src={src2.name}"
            )
        src = src2
    ok, sc = still_luma_ok(tmp)
    if not ok:
        img = ImageEnhance.Brightness(Image.open(tmp)).enhance(1.25)
        img.save(tmp, quality=92)
        ok, sc = still_luma_ok(tmp)
    # OCR area-ratio gate; on fail try next photo sources
    tries = 0
    while tries < min(8, len(photos)):
        _ok, _reason, _sc = passes_ocr_gate(tmp)
        if _ok:
            break
        tmp.unlink(missing_ok=True)
        tries += 1
        src = photos[(index - 1 + seed + tries * 3) % len(photos)]
        tmp = _render(src)
        geo, gsc = is_geometric_still(tmp)
        if geo:
            tmp.unlink(missing_ok=True)
            continue
        ok, sc = still_luma_ok(tmp)
        if not ok:
            img = ImageEnhance.Brightness(Image.open(tmp)).enhance(1.25)
            img.save(tmp, quality=92)
            ok, sc = still_luma_ok(tmp)
    else:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"make_fallback_still: OCR-area reject after retries {_reason}")
    tmp.replace(dest)
    print(
        f"[still] ok photoreal src={src.name} luma={sc.get('luma', 0):.1f} "
        f"glow_d={gsc.get('glow_delta', 0):.1f} -> {dest.name}",
        flush=True,
    )
    return dest
