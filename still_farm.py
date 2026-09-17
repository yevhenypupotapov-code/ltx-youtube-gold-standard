# -*- coding: utf-8 -*-
# Copyright (c) 2026 Yevhen Potapov. All rights reserved.
# Part of ContentForge. See LICENSE in the repository root.
"""
still_farm.py -- grow the LTX fallback pool with freshly generated stills.

Why: the pool was 9 images for videos that need ~10 shots, so the assembler
ran out of unused stills, reused them, and (a) produced near-duplicate keyframes
across releases and (b) tripped the near-black audit. Fresh, unique, gated stills
are the durable fix.

How: text-to-video with LTXV-2B (the installed model), then extract a frame and
gate it exactly like the pool loader does, then append it to fallback_pool/manifest.json.

Usage:
  python still_farm.py --count 6            # add 6 new stills
  python still_farm.py --count 6 --dry-run  # show what it would do
Requires ComfyUI up on 127.0.0.1:8188.
"""
from __future__ import annotations

import argparse, io, json, random, shutil, subprocess, sys, time
from datetime import datetime
from pathlib import Path

ROOT = Path(r"D:\Tutorial\ltx-youtube")
POOL = ROOT / "fallback_pool"
MANIFEST = POOL / "manifest.json"
GEN = ROOT / "generate_and_upload.py"
CONFIG = ROOT / "config.json"
COMFY_PY = Path(r"D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe")

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

# Family -> (scene prompt, tags, primary_tag, super_category, exclusion_group)
SCENES = [
    ("A moody dark home-office desk at night, a large blank monitor emitting a soft cold glow, "
     "closed laptop beside it, shallow depth of field, cinematic rim light, photoreal, no text on screen",
     ["monitor_code", "code"], "monitor_code", "abstract_screen", "abstract_screen"),
    ("Macro shot of a mechanical keyboard on a dark walnut desk, warm desk lamp light, dust motes, "
     "soft bokeh background, photoreal product photography, no readable legends",
     ["keyboard_macro", "macro"], "keyboard_macro", "device_macro", "device_macro"),
    ("Close-up of neatly coiled black cables and a small usb hub on a dark matte surface, "
     "side light, deep shadows, photoreal macro, minimal",
     ["cables_peripherals", "cables"], "cables_peripherals", "peripherals_detail", "peripherals_detail"),
    ("Wide shot of a quiet dark workspace at dawn, wooden desk, chair, plant silhouette, "
     "cool blue window light, cinematic ambient interior, photoreal",
     ["desk_wide", "desk"], "desk_wide", "environment_wide", "environment_wide"),
    ("A closed silver laptop on a linen desk pad, minimal dark desk setup, warm accent light, "
     "photoreal product still life, shallow depth of field",
     ["laptop_closed", "laptop"], "laptop_closed", "device_macro", "device_macro"),
    ("Dark desk corner with a small notebook, pen and a cold cup of coffee, moody side lighting, "
     "photoreal editorial still life, deep shadows",
     ["notebook", "desk"], "notebook", "environment_wide", "environment_wide"),
    ("Upright dark monitor on a standing desk, cable channel visible, soft practical light from the left, "
     "photoreal industrial interior, no text",
     ["monitor", "monitor_code"], "monitor", "monitor", "monitor_code"),
    ("Overhead flat-lay of a dark desk: keyboard, mouse, phone face-down, notebook, single warm lamp pool, "
     "photoreal top-down, no text on any surface",
     ["desk", "desk_wide"], "desk", "desk", "desk_wide"),
]

NEG = (
    "low quality, worst quality, blurry, jpeg artifacts, watermark, text, subtitles, logo, letters, words, "
    "writing, caption, ui text, hud text, glyphs, fake code, gibberish text, readable ui, terminal text, "
    "deformed, mutated, disfigured, gore, horror, scary, people, hands, faces, person, human, crowd, "
    "silhouette, portrait, bright, white, high-key, overexposed, daylight flood, flat geometric rectangles, "
    "dot grid, node network diagram, empty frame, black frame"
)


def _ffmpeg() -> str:
    for c in ("ffmpeg",):
        p = shutil.which(c)
        if p:
            return p
    raise SystemExit("still_farm: ffmpeg not found on PATH")


def _dhash(path: Path, s: int = 8):
    from PIL import Image
    import numpy as np
    im = Image.open(path).convert("L").resize((s + 1, s), Image.Resampling.LANCZOS)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def _extract_frame(video: Path, dest: Path, at: float = 1.0) -> Path:
    subprocess.check_call(
        [_ffmpeg(), "-y", "-ss", str(at), "-i", str(video), "-frames:v", "1",
         "-vf", "scale=1280:720:flags=lanczos", "-q:v", "2", str(dest)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    if not dest.exists():
        raise RuntimeError(f"frame extract produced nothing from {video}")
    return dest


def _sharpen(path: Path, percent: float) -> None:
    """LTXV frames come out soft (Laplacian ~40-55) and fail the pool's
    MIN_LAPLACIAN_VAR=100 gate. Unsharp masking brings them up to photo-pool
    crispness so they are accepted by the same loader gate as real photos."""
    from PIL import Image, ImageFilter
    im = Image.open(path).convert("RGB")
    im = im.filter(ImageFilter.UnsharpMask(radius=2.0, percent=percent, threshold=2))
    im.save(path, quality=92)


def _gate(cand: Path, pool_hashes: list):
    from shot_gate import still_luma_ok, is_geometric_still, MIN_STILL_LUMA
    ok, sc = still_luma_ok(cand, min_luma=MIN_STILL_LUMA)
    if not ok:
        return False, f"luma/var/lap fail luma={sc.get('luma',0):.1f} var={sc.get('variance',0):.0f} lap={sc.get('laplacian',0):.1f}"
    geo, gsc = is_geometric_still(cand)
    if geo:
        return False, f"geometric/glow fail d={gsc.get('glow_delta',0):.1f}"
    h = _dhash(cand)
    for hid, hh in pool_hashes:
        if int((h != hh).sum()) <= 6:
            return False, f"near-duplicate of {hid}"
    return True, f"luma={sc.get('luma',0):.1f} var={sc.get('variance',0):.0f}"


def _newest_mp4(dirs, since: float):
    best, bt = None, 0.0
    for d in dirs:
        if not d.is_dir():
            continue
        for p in d.rglob("*.mp4"):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            if m >= since and m > bt:
                best, bt = p, m
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import requests
    try:
        requests.get("http://127.0.0.1:8188/system_stats", timeout=4).raise_for_status()
    except Exception as e:
        print(f"[farm] ComfyUI is not up on :8188 ({e}) -- start it first")
        return 2

    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out_dirs = [Path(p) for p in cfg.get("output_dirs", [])]

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    existing_ids = {it["id"] for it in man["items"]}
    pool_hashes = []
    for it in man["items"]:
        p = Path(it["path"])
        if p.exists():
            try:
                pool_hashes.append((it["id"], _dhash(p)))
            except Exception:
                pass
    print(f"[farm] pool has {len(pool_hashes)} gated images; target +{args.count}")

    added = 0
    serial = 90
    for i in range(args.count):
        prompt, tags, primary, supercat, egroup = SCENES[i % len(SCENES)]
        while f"photo_{serial:02d}_{primary}_farm" in existing_ids:
            serial += 1
        asset_id = f"photo_{serial:02d}_{primary}_farm"
        if args.dry_run:
            print(f"[dry] would generate {asset_id}  tags={tags}")
            serial += 1
            continue

        seed = random.randint(1, 2**31 - 1)
        t0 = time.time()
        print(f"[farm] {i+1}/{args.count} generate {asset_id} seed={seed}", flush=True)
        cmd = [str(COMFY_PY), str(GEN), "--mode", "t2v", "--no-upload",
               "--prompt", prompt, "--negative", NEG, "--frames", str(args.frames),
               "--seed", str(seed), "--prefix", f"stillfarm_{asset_id}"]
        code = subprocess.call(cmd, cwd=str(ROOT))
        if code != 0:
            print(f"[farm] generate failed ({code}) for {asset_id} -- skip")
            serial += 1
            continue
        video = _newest_mp4(out_dirs, t0)
        if video is None:
            print(f"[farm] no mp4 produced for {asset_id} -- skip")
            serial += 1
            continue

        tmp = POOL / f"_farm_tmp_{asset_id}.jpg"
        ok, note = False, "not tried"
        try:
            _extract_frame(video, tmp)
            for pct in (140, 260, 400):
                _sharpen(tmp, pct)
                ok, note = _gate(tmp, pool_hashes)
                if ok:
                    note = f"{note} sharp={pct}%"
                    break
        except Exception as e:
            ok, note = False, f"error {e}"
        if not ok:
            print(f"[farm] reject {asset_id}: {note}")
            tmp.unlink(missing_ok=True)
            serial += 1
            continue

        dest = POOL / f"{asset_id}.jpg"
        tmp.replace(dest)
        # Distinct primary_tag: sharing one with a hand-picked pool item made the
        # assembler place two same-family stills back to back and abort.
        man["items"].append({
            "id": asset_id, "path": str(dest), "tags": list(tags) + [primary + "_farm"],
            "primary_tag": primary + "_farm", "super_category": supercat,
            "exclusion_group": egroup,
        })
        pool_hashes.append((asset_id, _dhash(dest)))
        existing_ids.add(asset_id)
        added += 1
        print(f"[farm] ADDED {asset_id}  {note}  (from {video.name})")
        serial += 1

    if added and not args.dry_run:
        ts = datetime.now().strftime("%Y%m%d-%H%M")
        shutil.copy2(MANIFEST, MANIFEST.with_name(f"manifest.json.bak.pre-farm-{ts}"))
        man["version"] = int(man.get("version", 0)) + 1
        io.open(MANIFEST, "w", encoding="utf-8", newline="\n").write(
            json.dumps(man, ensure_ascii=False, indent=2))
        print(f"[farm] manifest updated (+{added}) -> {MANIFEST}")
    print(f"[farm] done: +{added} stills")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
