from __future__ import annotations



import argparse
import os

import subprocess

import sys

import time

from pathlib import Path



ROOT = Path(__file__).resolve().parent

if str(ROOT) not in sys.path:

    sys.path.insert(0, str(ROOT))

FACTORY = Path(os.environ.get("LTX_FACTORY", str(ROOT / "factory"))).expanduser()
COMFY_PY = Path(os.environ.get("LTX_COMFY_PY", "python")).expanduser()



def find_ffmpeg() -> str:
    """Resolve ffmpeg: LTX_FFMPEG env, then PATH, then common install locations."""
    import shutil
    env = os.environ.get("LTX_FFMPEG", "").strip()
    if env and Path(env).exists():
        return env
    which = shutil.which("ffmpeg")
    if which:
        return which
    candidates = [
        Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
        Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
        Path("/usr/bin/ffmpeg"),
        Path("/usr/local/bin/ffmpeg"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "ffmpeg"

FFMPEG = find_ffmpeg()



def ffmpeg_bin() -> str:

    return str(FFMPEG)





SAFETY_NEGATIVE = (

    "low quality, worst quality, blurry, jpeg artifacts, watermark, text, subtitles, logo, "

    "letters, words, writing, caption, ui text, hud text, glyphs, glyphs on screen, "

    "readable ui, interface chrome, menu bar, icons with labels, fake code, gibberish text, "

    "scribbles on monitor, terminal text, keyboard legends readable, "

    "any text on screen, any glyphs on monitor, mark-like patterns on display, "

    "sharp monitor UI, crisp desktop icons, window chrome, taskbar, cursor arrow, "

    "blank monitor preferred, out-of-focus screen, blurred display, soft glowing blank screen, "

    "no readable characters, no fake code, no terminal glyphs, "

    "deformed, mutated, disfigured, bad anatomy, extra fingers, missing limbs, "

    "severed head, disembodied head, gore, horror, creepy, uncanny, nonsense geometry, "

    "flat geometric rectangles, teal white rectangles, dot grid, node network diagram, "

    "abstract schematic, connected nodes, LED grid board, "

    "empty frame, black frame, morphing face, melted face, duplicate faces, twins, "

"people, hands, faces, person, human, portrait, selfie, typing hands, "
    "bright, white, high-key, overexposed, bright office, white desk, daylight flood, "
    "ui, hud, dashboard chrome, "
        "green skin, wax face, mask straps, human face close-up, portrait, people faces"

)



# LTX length must be 8n+1. 81 ≈ 3.4s @24fps; 97 ≈ 4.0s

DEFAULT_LONG_FRAMES = 81



MOTION = [
    "slow cinematic pan left over dark tech desk, visible camera drift, subtle parallax",
    "slow cinematic pan right over dark tech desk, visible camera drift, subtle parallax",
    "gentle camera push-in, subtle parallax, dark moody lighting",
    "slow pull back revealing more of the dark desk, subtle parallax",
    "slight orbit around the subject, cinematic, dark tech workspace",
    "camera eases closer, slow cinematic motion, same dark scene",
]





def ffmpeg_bin() -> str:

    return str(FFMPEG)



def run_slides(argv: list[str]) -> int:

    script = FACTORY / "video_factory.py"

    if not script.exists():

        raise SystemExit(f"missing {script}")

    cmd = [sys.executable, str(script), *argv]

    print(f"[slides] {' '.join(cmd)}", flush=True)

    return subprocess.call(cmd, cwd=str(FACTORY))





def run_ltx(args: argparse.Namespace) -> int:

    gen = ROOT / "generate_and_upload.py"

    cmd = [str(COMFY_PY if COMFY_PY.exists() else sys.executable), str(gen)]

    cmd += ["--mode", args.mode]

    if args.prompt:

        cmd += ["--prompt", args.prompt]

    if args.image:

        cmd += ["--image", str(args.image)]

    if args.no_upload:

        cmd.append("--no-upload")

    else:

        cmd += ["--privacy", args.privacy]

        if args.title:

            cmd += ["--title", args.title]

    if args.frames:

        cmd += ["--frames", str(args.frames)]

    print(f"[ltx] {' '.join(cmd)}", flush=True)

    return subprocess.call(cmd, cwd=str(ROOT))





def _comfy_py() -> str:

    return str(COMFY_PY if COMFY_PY.exists() else sys.executable)





def last_frame(video: Path, dest: Path) -> Path:

    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [

        ffmpeg_bin(), "-y", "-sseof", "-0.08", "-i", str(video),

        "-frames:v", "1", "-q:v", "2", str(dest),

    ]

    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not dest.exists() or dest.stat().st_size < 200:

        raise SystemExit(f"failed to extract last frame from {video}")

    return dest





def concat_videos(clips: list[Path], dest: Path) -> Path:

    dest.parent.mkdir(parents=True, exist_ok=True)

    listing = dest.with_suffix(".concat.txt")

    lines = []

    for clip in clips:

        path = str(clip.resolve()).replace("\\", "/")

        lines.append(f"file '{path}'")

    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [

        ffmpeg_bin(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),

        "-c:v", "libx264", "-preset", "fast", "-crf", "20",

        "-pix_fmt", "yuv420p", "-r", "24", "-an",

        "-movflags", "+faststart", str(dest),

    ]

    subprocess.check_call(cmd)

    if not dest.exists() or dest.stat().st_size < 1000:

        raise SystemExit(f"concat failed: {dest}")

    return dest





def generate_clip(

    *,

    mode: str,

    prompt: str,

    prefix: str,

    frames: int,

    image: Path | None = None,

    negative: str | None = None,

) -> Path:

    from generate_and_upload import load_config, newest_mp4



    gen = ROOT / "generate_and_upload.py"

    cmd = [

        _comfy_py(), str(gen),

        "--mode", mode,

        "--prompt", prompt,

        "--negative", negative or SAFETY_NEGATIVE,

        "--no-upload",

        "--prefix", prefix,

        "--frames", str(frames),

    ]

    if image is not None:

        cmd += ["--image", str(image)]

    print(f"[long] {mode} {prefix}", flush=True)

    started = time.time()

    code = subprocess.call(cmd, cwd=str(ROOT))

    if code != 0:

        raise SystemExit(f"long: generate_and_upload failed ({code}) for {prefix}")

    cfg = load_config(ROOT / "config.json")

    dirs = [Path(p) for p in cfg.get("output_dirs", [])]

    video = newest_mp4(dirs, started)

    if video is None:

        raise SystemExit(f"long: no mp4 after {prefix}")

    print(f"[long] clip {video}", flush=True)

    return video





def run_hybrid(args: argparse.Namespace) -> int:

    from yt import CHANNEL_HANDLE, CHANNEL_URL, upload_video

    if str(FACTORY) not in sys.path:

        sys.path.append(str(FACTORY))

    import video_factory as vf  # noqa: E402



    topic = vf.pick_topic(args.topic, args.lang)

    content = vf.generate_ltx_plan(

        topic, args.lang, args.minutes, args.model, segments=1

    )

    title = content.get("title") or topic[:90]

    shots = content.get("shots") or []

    visual = (

        (shots[0].get("visual") if shots else None)

        or (

            f"Cinematic 16:9 footage illustrating: {title}. "

            "Photoreal motion, natural lighting, no text, no watermark, no subtitles."

        )

    )

    gen = ROOT / "generate_and_upload.py"

    py = _comfy_py()

    out_prefix = "video/LTX_hybrid"

    cmd = [

        py, str(gen),

        "--mode", "t2v",

        "--prompt", visual,

        "--title", title,

        "--no-upload",

        "--prefix", out_prefix,

        "--frames", str(args.frames or 49),

    ]

    print(f"[hybrid] LTX visual for: {title}", flush=True)

    started = time.time()

    code = subprocess.call(cmd, cwd=str(ROOT))

    if code != 0:

        return code



    from generate_and_upload import load_config, newest_mp4

    cfg = load_config(ROOT / "config.json")

    dirs = [Path(p) for p in cfg.get("output_dirs", [])]

    video = newest_mp4(dirs, started)

    if video is None:

        raise SystemExit("hybrid: LTX finished but no mp4 found")

    print(f"[hybrid] video {video}", flush=True)



    script = str(content.get("script") or title or topic or "").strip()

    from add_audio import add_voiceover

    video = add_voiceover(video, script, args.lang)



    if args.no_upload:

        vf.remember(topic, None, None)

        print("[hybrid] skipped YouTube", flush=True)

        return 0



    desc = content.get("description") or title

    if CHANNEL_HANDLE not in desc:

        desc += f"\n\n{CHANNEL_HANDLE}\n{CHANNEL_URL}"

    upload_video(

        video,

        title=title,

        description=desc,

        tags=list(content.get("tags") or ["ai", "ltx-video"]),

        privacy=args.privacy,

        topic=topic,

        engine="hybrid",

    )

    return 0





def _script_bundle(args: argparse.Namespace) -> tuple[str, str, str, dict]:

    """Return topic, title, spoken script, and LTX R2V plan dict."""

    topic = args.topic or args.prompt or "Local AI on a laptop GPU"

    title = args.title or topic[:90]

    script = ""

    content: dict = {}

    if str(FACTORY) not in sys.path:

        sys.path.append(str(FACTORY))

    try:

        import video_factory as vf  # noqa: E402

        import json as _json

        topic = vf.pick_topic(args.topic, args.lang)

        n = max(8, int(getattr(args, "segments", None) or 10))

        content = vf.generate_ltx_plan(

            topic, args.lang, min(args.minutes, 4), args.model, segments=n

        )

        title = content.get("title") or topic[:90]

        script = str(content.get("script") or "").strip()

        plan_path = ROOT / "long_work" / "r2v_plan.json"

        plan_path.parent.mkdir(exist_ok=True)

        plan_path.write_text(

            _json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8"

        )

        print(f"[long] R2V plan -> {plan_path}", flush=True)

    except Exception as exc:

        print(f"[long] factory R2V plan skipped ({exc})", flush=True)

    if not script:

        if args.lang == "ru":

            script = (

                f"{title}. Короткий тест длинного ролика: несколько сцен подряд, "

                "склеенных с последнего кадра, и живая озвучка поверх."

            )

        else:

            script = (

                f"{title}. A short test of a longer clip: several shots chained "

                "from the last frame, with a voiceover on top."

            )

    return topic, title, script, content







def run_long(args: argparse.Namespace) -> int:

    from yt import CHANNEL_HANDLE, CHANNEL_URL, upload_video

    from add_audio import add_voiceover, tts_to_mp3, probe_duration, mux_fit

    from video_assembler import (

        ShotCandidate,

        PipelineAbortError,

        assemble_video,

        load_assembler_config,

        MAX_STILL_HOLD_SEC,

    )



    topic, title, script, content = _script_bundle(args)

    shots = list(content.get("shots") or []) if isinstance(content, dict) else []

    base_visual = (

        args.prompt

        or (

            f"Cinematic 16:9 object-and-space footage for: {title}. "

            "Photoreal, natural lighting, prefer hands/objects/spaces over faces, "

            "no text, no watermark, no subtitles, no logos."

        )

    )

    frames = int(args.frames or DEFAULT_LONG_FRAMES)

    if args.segments:

        n = max(2, int(args.segments))

    else:

        words = len(str(script).split())

        n = max(8, min(12, words // 12 or 8))

    if len(shots) >= 8:

        n = max(n, min(12, len(shots)))

    elif len(shots) >= 2:

        n = max(n, len(shots))

    work = ROOT / "long_work"

    work.mkdir(exist_ok=True)

    gate_dir = work / "gates"

    gate_dir.mkdir(exist_ok=True)

    assemble_dir = work / "assemble"

    assemble_dir.mkdir(exist_ok=True)



    cfg_asm = load_assembler_config()

    max_hold = float(cfg_asm.get("max_still_hold_sec", MAX_STILL_HOLD_SEC))

    strict = bool(getattr(args, "strict_assemble", False))



    accepted: list[ShotCandidate] = []

    accepted_kfs: list[Path] = []

    last_img: Path | None = None

    prev_hist = None

    shot_sec = min(max(3.5, frames / 24.0), max_hold)

    # ~25-30% of beats get LTX attempts: 1 LTX per 3-4 stills (1-based)
    ltx_slots = {i for i in range(1, n + 1) if (i % 4) == 1}  # e.g. [1,5,9] for n=10 (~30%)

    print(

        f"[long] {n} beats, still-first, LTX inserts at {sorted(ltx_slots)} "

        f"(~{shot_sec:.1f}s cap={max_hold}s) title={title}",

        flush=True,

    )

    import json as _json

    from shot_gate import (

        evaluate_clip,

        make_fallback_still,

        still_to_kenburns_clip,

    )

    import numpy as np



    gate_log = []

    beat_texts: list[str] = []

    for i in range(n):

        shot = shots[i] if i < len(shots) else None

        visual = (shot.get("visual") if isinstance(shot, dict) else None) or base_visual

        script_beat = ""

        beat_label = f"beat{i+1}"

        if isinstance(shot, dict):

            script_beat = str(shot.get("script_beat") or "").strip()

            beat_label = str(shot.get("beat") or beat_label)

            visual = str(visual)

        beat_texts.append(script_beat or beat_label)

        visual = (

            f"{visual} Dark tech desk, moody cinematic lighting, objects and spaces only. "

            "No people, no hands, no faces, no twins, no portraits, no readable text on screens, "

            "no glyphs, no UI chrome, no fake code, blank or heavily blurred monitors only, "

            "solid soft glow screens OK, no logos, no subtitles, no bright white flood."

        )

        motion = MOTION[i % len(MOTION)]

        prompt = visual if visual.rstrip().endswith(".") else f"{visual}."

        prompt = (

            f"{prompt} {motion}. "

            "Slow cinematic pan over dark tech desk, subtle parallax. "
            "Monitors are blank soft glowing screens, out of focus, no UI, no text, no glyphs, "

            "no icons, photoreal dark desk only, no near-black empty frames, no bright white."

        )

        if script_beat:

            prompt = f"{prompt} Narration beat: {script_beat[:100]}"



        use_ltx = (i + 1) in ltx_slots

        clip: Path | None = None

        src = "still"

        if use_ltx:

            # Up to 2 candidates per LTX slot when first is OCR/UI-rejected (cheap retry)

            accepted_ltx = False

            for cand in (1, 2):

                prefix = f"video/LTX_long_{i+1:02d}" + (f"_c{cand}" if cand > 1 else "")

                try:

                    cand_prompt = prompt

                    if cand > 1:

                        cand_prompt = (

                            prompt + " Extreme soft focus on any screens, completely blank monitors, "

                            "no marks, no patterns on displays."

                        )

                    if last_img is not None and last_img.exists() and i % 2 == 1 and cand == 1:

                        clip = generate_clip(

                            mode="i2v", prompt=cand_prompt, prefix=prefix, frames=frames, image=last_img

                        )

                    else:

                        clip = generate_clip(mode="t2v", prompt=cand_prompt, prefix=prefix, frames=frames)

                    gdir = gate_dir / f"shot_{i+1:02d}" / f"c{cand}"

                    result = evaluate_clip(

                        clip,

                        script_beat=script_beat or beat_label,

                        visual_brief=visual,

                        work_dir=gdir,

                        prev_hist=prev_hist,

                    )

                    gate_log.append({"i": i + 1, "src": "ltx", "cand": cand, **result.to_dict()})

                    print(

                        f"[gate] shot {i+1} LTX cand={cand} accepted={result.accepted} reasons={result.reasons} "

                        f"scores={{m:{result.scores.get('motion',0):.1f} g:{result.scores.get('green_skin',0):.3f} "

                        f"ocr:{result.scores.get('ocr_chars', result.scores.get('ocr_gibber_hits',0)):.0f}}}",

                        flush=True,

                    )

                    if result.accepted:

                        src = "ltx"

                        accepted_ltx = True

                        if result.keyframe:

                            accepted_kfs.append(Path(result.keyframe))

                        hist_file = gdir / "last_hist.npy"

                        if hist_file.exists():

                            prev_hist = np.load(hist_file)

                        accepted.append(

                            ShotCandidate(

                                asset_id=f"ltx_{i+1:02d}",

                                path=clip,

                                is_fallback=False,

                                tags=["workspace", "desk", "monitor_glow"],

                                motion_score=float(result.scores.get("motion") or 0.0),

                                script_beat=script_beat or beat_label,

                            )

                        )

                        break

                    else:

                        clip = None

                        # Retry only when OCR/UI / style reject — skip 2nd try on hard uncanny/horror

                        hard = any(

                            str(r).startswith(("horror", "uncanny", "face", "tech:"))

                            for r in (result.reasons or [])

                        )

                        if hard:

                            print(

                                f"[gate] shot {i+1} LTX hard-reject {result.reasons} — no 2nd cand",

                                flush=True,

                            )

                            break

                        if cand == 1:

                            print(

                                f"[gate] shot {i+1} LTX rejected ({result.reasons}) — trying 2nd candidate",

                                flush=True,

                            )

                except Exception as exc:

                    print(f"[gate] shot {i+1} LTX cand={cand} failed ({exc})", flush=True)

                    gate_log.append({"i": i + 1, "src": "ltx_error", "cand": cand, "error": str(exc)})

                    clip = None

            if not accepted_ltx:

                clip = None

                print(

                    f"[gate] shot {i+1} LTX rejected -> assembler will use diversified fallback pool",

                    flush=True,

                )



        if clip is None:

            # Unique per-beat still candidate (diversified); KenBurns capped at max_hold

            still = work / f"fallback_{i+1:02d}.jpg"

            make_fallback_still(still, beat=beat_label, title=title, index=i + 1)

            from shot_gate import still_luma_ok

            ok_still, lsc = still_luma_ok(still)

            if not ok_still:

                print(

                    f"[gate] shot {i+1} still too dark/flat luma={lsc.get('luma',0):.1f} — regenerating brighter",

                    flush=True,

                )

                make_fallback_still(

                    still, beat=beat_label + "|bright", title=title + "|lit", index=i + 101,

                )

                ok_still, lsc = still_luma_ok(still)

            if not ok_still:

                print(

                    f"[gate] shot {i+1} still FAILED luma gate luma={lsc.get('luma',0):.1f} — assembler pool must cover",

                    flush=True,

                )

            clip = work / f"fallback_{i+1:02d}.mp4"

            still_to_kenburns_clip(still, clip, seconds=min(shot_sec, max_hold), max_still_hold_sec=max_hold)

            accepted_kfs.append(still)

            src = "still"

            gate_log.append({

                "i": i + 1, "src": "still_fallback", "accepted": bool(ok_still),

                "is_fallback": True, "luma": float(lsc.get("luma", 0)),

            })

            print(

                f"[gate] shot {i+1} using still+KenBurns fallback "

                f"(unique layout, hold<={max_hold}s, luma={lsc.get('luma',0):.1f})",

                flush=True,

            )

            accepted.append(

                ShotCandidate(

                    asset_id=f"still_{i+1:02d}",

                    path=clip,

                    is_fallback=True,

                    tags=["workspace", "desk", "abstract_tech", "laptop"],

                    motion_score=0.05,

                    script_beat=script_beat or beat_label,

                )

            )



        try:

            last_img = last_frame(clip, work / f"last_{i+1:02d}.jpg")

        except Exception:

            last_img = accepted_kfs[-1] if accepted_kfs else last_img

        print(f"[long] beat {i+1}/{n} src={src} -> {clip.name}", flush=True)



    (gate_dir / "gate_log.json").write_text(

        _json.dumps(gate_log, ensure_ascii=False, indent=2), encoding="utf-8"

    )



    # --- TTS early so assembler covers real audio duration ---

    out_dir = Path(os.environ.get("LTX_OUTPUT_DIR", str(ROOT / "output"))).expanduser()

    out_dir.mkdir(parents=True, exist_ok=True)

    tts_path = out_dir / "LTX_long_concat_tts.mp3"

    try:

        tts_to_mp3(script, tts_path, args.lang)

        audio_dur = probe_duration(tts_path)

    except Exception as exc:

        print(f"[long] TTS probe fallback estimate ({exc})", flush=True)

        wps = 2.2 if args.lang == "ru" else 2.5

        audio_dur = max(8.0, len(str(script).split()) / wps)

        tts_path = None



    print(f"[assemble] audio_duration={audio_dur:.2f}s approved_candidates={len(accepted)}", flush=True)



    # Prefer non-fallback LTX as approved inputs; stills are candidates but assembler

    # prefers pool diversification + LTX. Pass LTX-approved first.

    approved_ltx = [c for c in accepted if not c.is_fallback]

    # Also pass unique still candidates as secondary (tagged fallback)

    approved_for_asm = approved_ltx  # empty gates -> pool stills only (NOT random LTX)

    for c in approved_ltx:

        print(

            f"[long] LTX_APPROVED_FOR_CONCAT asset={c.asset_id} path={c.path.name} "

            f"motion={c.motion_score:.2f} — must appear in assemble timeline (not silent still-replace)",

            flush=True,

        )

    if not approved_ltx:

        print("[long] warning: zero LTX approved — timeline will be pool stills only", flush=True)



    stitched = out_dir / "LTX_long_concat.mp4"

    thumb = out_dir / "LTX_long_thumb.jpg"

    quality_compromised = False

    try:

        asm = assemble_video(

            approved=approved_for_asm,

            audio_duration=audio_dur,

            beat_texts=beat_texts or [title],

            out_concat=stitched,

            thumb_dest=thumb,

            work_dir=assemble_dir,

            strict=strict,

            soft_fail=(not strict),

        )

    except PipelineAbortError as exc:

        print(f"[assemble] CRITICAL hard-abort: {exc}", flush=True)

        print("[long] refusing to mux/upload a stretched stub", flush=True)

        return 2



    if not asm.ok or asm.quality_compromised:

        quality_compromised = True

        print(

            f"[assemble] CRITICAL soft-fail quality_compromised=true reason={asm.reason}",

            flush=True,

        )

        print("[long] skip YouTube upload; no stub mux", flush=True)

        (assemble_dir / "quality_compromised.json").write_text(

            _json.dumps({"quality_compromised": True, "reason": asm.reason}, indent=2),

            encoding="utf-8",

        )

        return 3



    print(f"[long] concat {stitched} clips={len(asm.clips)} unique={asm.unique_assets}", flush=True)



    # Mux with pre-made TTS (no long last-frame pad expected if coverage matches)

    if tts_path and tts_path.exists():

        video = out_dir / "LTX_long_concat_vo.mp4"

        mux_fit(stitched, tts_path, video)

    else:

        video = add_voiceover(stitched, script, args.lang)

    print(f"[long] with voice {video}", flush=True)



    if args.no_upload or quality_compromised:

        print("[long] skipped YouTube", flush=True)

        return 0



    desc = ""

    tags = ["ai", "ltx-video", "long"]

    if isinstance(content, dict):

        desc = str(content.get("description") or "")

        tags = list(content.get("tags") or tags)

    if not desc:

        desc = title

    if CHANNEL_HANDLE not in desc:

        desc += f"\\n\\n{CHANNEL_HANDLE}\\n{CHANNEL_URL}"

    upload_video(

        video,

        title=title,

        description=desc,

        tags=tags,

        privacy=args.privacy,

        topic=topic,

        engine="long",

    )

    return 0





def main() -> int:

    p = argparse.ArgumentParser(description="LTX YouTube factory: slides / LTX / hybrid / long")

    p.add_argument("--engine", choices=["slides", "ltx", "hybrid", "long"], default="hybrid")

    p.add_argument("--topic", default=None)

    p.add_argument("--prompt", default=None, help="Direct LTX prompt (engine=ltx|long)")

    p.add_argument("--lang", default="en", choices=["ru", "en"])

    p.add_argument("--minutes", type=int, default=6)

    p.add_argument("--model", default="ltx-factory:latest")

    p.add_argument("--mode", choices=["t2v", "i2v"], default="t2v")

    p.add_argument("--image", type=Path, default=None)

    p.add_argument("--frames", type=int, default=None)

    p.add_argument("--segments", type=int, default=10, help="Unique shots for --engine long (no looping)")

    p.add_argument("--title", default=None)

    p.add_argument("--privacy", default="private", choices=["private", "unlisted", "public"])

    p.add_argument("--no-upload", action="store_true")

    p.add_argument(

        "--strict-assemble",

        action="store_true",

        help="Hard PipelineAbortError if timeline cannot cover audio QC",

    )

    args, rest = p.parse_known_args()



    if args.engine == "slides":

        extra = rest[:]

        if args.topic:

            extra += ["--topic", args.topic]

        extra += ["--lang", args.lang, "--minutes", str(max(4, min(8, args.minutes)))]

        extra += ["--privacy", args.privacy]

        if args.no_upload:

            extra.append("--no-upload")

        return run_slides(extra)



    if args.engine == "ltx":

        if not args.prompt:

            raise SystemExit("--engine ltx needs --prompt")

        return run_ltx(args)



    if args.engine == "long":

        return run_long(args)



    return run_hybrid(args)





if __name__ == "__main__":

    sys.exit(main())







