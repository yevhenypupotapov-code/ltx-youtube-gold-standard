#!/usr/bin/env python3
"""Assembly gates for LTX factory shots.

Order: tech -> horror/uncanny -> OCR garbage -> motion -> style vs previous.
Reject => caller must use still+KenBurns fallback (no faces, no readable UI).
"""
from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

def _ffmpeg_bin() -> str:
    import os, shutil
    env = os.environ.get("LTX_FFMPEG", "").strip()
    if env and Path(env).exists():
        return env
    which = shutil.which("ffmpeg")
    if which:
        return which
    return "ffmpeg"

FFMPEG = Path(_ffmpeg_bin())  # may be literal "ffmpeg" if not on PATH


def _ffmpeg() -> str:
    return str(FFMPEG) if FFMPEG.exists() else "ffmpeg"


def _ffprobe() -> str:
    ff = Path(_ffmpeg())
    p = ff.with_name("ffprobe.exe" if ff.suffix.lower() == ".exe" else "ffprobe")
    return str(p) if p.exists() else "ffprobe"


@dataclass
class GateResult:
    accepted: bool
    reasons: list[str]
    scores: dict[str, float]
    keyframe: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def extract_keyframes(video: Path, dest_dir: Path, count: int = 3) -> list[Path]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    # sample at 20%, 50%, 80%
    fracs = [0.2, 0.5, 0.8][:count]
    out: list[Path] = []
    # duration
    raw = subprocess.check_output(
        [_ffprobe(), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(video)],
        text=True, encoding="utf-8", errors="replace",
    )
    dur = float(json.loads(raw).get("format", {}).get("duration") or 2.0)
    for i, f in enumerate(fracs):
        t = max(0.05, min(dur - 0.05, dur * f))
        dest = dest_dir / f"{video.stem}_kf{i}.jpg"
        subprocess.check_call(
            [_ffmpeg(), "-y", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1", "-q:v", "3", str(dest)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if dest.exists() and dest.stat().st_size > 500:
            out.append(dest)
    return out


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    img = img.resize((640, 360), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32)


# Near-black desk stubs (~17-22 mean) look like a broken/noisy screen on YT.
# Require visibly lit frames (office/desk mean typically 45-140).
MIN_STILL_LUMA = 40.0
MAX_STILL_LUMA = 245.0  # absolute hard ceiling (near-white)
# Dark-office palette gate (pool admit / LTX bright-office reject)
# Empirically: neon-purple dark laptop ~47 passes; MacBook-on-white-desk ~165-240 fails.
MAX_DARK_OFFICE_MEAN_LUMA = 100.0  # aligned with NEGATIVE_MAX_AVG_LUMA
MAX_DARK_OFFICE_P90_LUMA = 220.0
MAX_SKIN_RATIO_POOL = 0.12  # people/hands; wood tones usually <0.10
MAX_SKIN_RATIO_LTX = 0.10
MIN_STILL_VAR = 120.0
MIN_LAPLACIAN_VAR = 100.0  # OpenCV Laplacian.var(); flat lids/noise fail


def gate_tech(frames: list[np.ndarray]) -> tuple[bool, str, dict[str, float]]:
    if not frames:
        return False, "tech:no_frames", {}
    means = [float(f.mean()) for f in frames]
    vars_ = [float(f.var()) for f in frames]
    mean_luma = float(np.mean(means))
    mean_var = float(np.mean(vars_))
    scores = {"luma": mean_luma, "variance": mean_var}
    if mean_luma < MIN_STILL_LUMA:
        return False, "tech:too_dark", scores
    if mean_luma > MAX_STILL_LUMA:
        return False, "tech:blown_out", scores
    if mean_var < MIN_STILL_VAR:
        return False, "tech:flat_low_detail", scores
    # Extreme film-grain on near-black = "noisy broken screen"
    if mean_luma < 55.0 and mean_var > 4500.0:
        return False, "tech:dark_noise", scores
    return True, "", scores


def laplacian_variance(path: Path) -> float:
    """OpenCV Laplacian.var() structure score (grain can fool mean luma/variance)."""
    try:
        import cv2  # type: ignore
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0
        return float(cv2.Laplacian(img, cv2.CV_64F).var())
    except Exception:
        try:
            im = Image.open(path).convert("L")
            arr = np.asarray(im, dtype=np.float32)
            # numpy fallback 4-neighbor laplacian
            up = np.roll(arr, 1, 0)
            dn = np.roll(arr, -1, 0)
            lf = np.roll(arr, 1, 1)
            rt = np.roll(arr, -1, 1)
            lap = up + dn + lf + rt - 4.0 * arr
            return float(lap.var())
        except Exception:
            return 0.0


def still_luma_ok(
    path: Path,
    min_luma: float | None = None,
    min_var: float | None = None,
    min_laplacian: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """Reject near-black / flat / low-structure / extreme-noise stills before KenBurns or pool use."""
    min_luma = float(MIN_STILL_LUMA if min_luma is None else min_luma)
    min_var = float(MIN_STILL_VAR if min_var is None else min_var)
    min_lap = float(MIN_LAPLACIAN_VAR if min_laplacian is None else min_laplacian)
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im).astype(np.float32)
    except Exception:
        return False, {"luma": 0.0, "variance": 0.0, "laplacian": 0.0, "error": 1.0}
    luma = float(arr.mean())
    var = float(arr.var())
    lap = float(laplacian_variance(path))
    scores = {"luma": luma, "variance": var, "laplacian": lap}
    if luma < min_luma:
        return False, scores
    if luma > MAX_STILL_LUMA:
        return False, scores
    if var < min_var:
        return False, scores
    if lap < min_lap:
        return False, scores
    if luma < 55.0 and var > 4500.0:
        return False, scores
    return True, scores



def mean_luma_rgb(arr) -> float:
    """Rec.601 luma mean on float RGB array."""
    import numpy as np
    a = np.asarray(arr, dtype=np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    return float((0.299 * r + 0.587 * g + 0.114 * b).mean())


def skin_ratio_rgb(arr) -> float:
    import numpy as np
    a = np.asarray(arr, dtype=np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & ((r - b) > 15)
    return float(skin.mean())


def dark_office_luma_ok(path: Path) -> tuple[bool, dict[str, float]]:
    """Reject high-key / bright white desks for dark-office palette.

    Tuned so bright MacBook-on-white-desk fails and neon-purple dark laptop passes.
    """
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im).astype(np.float32)
    except Exception:
        return False, {"mean_luma": 0.0, "p90_luma": 0.0, "error": 1.0}
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mean_l = float(gray.mean())
    p90 = float(np.percentile(gray, 90))
    scores = {"mean_luma": mean_l, "p90_luma": p90, "luma": mean_l}
    if mean_l > float(MAX_DARK_OFFICE_MEAN_LUMA):
        return False, scores
    if p90 > float(MAX_DARK_OFFICE_P90_LUMA) and mean_l > 90.0:
        return False, scores
    return True, scores


def people_hands_face_ok(path: Path, *, max_skin: float | None = None) -> tuple[bool, dict[str, float]]:
    """Reject stills with hands / arms / faces / people (skin heuristic + lower-half focus)."""
    max_skin = float(MAX_SKIN_RATIO_POOL if max_skin is None else max_skin)
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im).astype(np.float32)
    except Exception:
        return False, {"skin_ratio": 1.0, "skin_lo": 1.0, "error": 1.0}
    h = arr.shape[0]
    skin = skin_ratio_rgb(arr)
    skin_lo = skin_ratio_rgb(arr[h // 2 :, :, :])
    # center strip often captures typing hands
    y0, y1 = int(h * 0.35), int(h * 0.95)
    x0, x1 = int(arr.shape[1] * 0.15), int(arr.shape[1] * 0.85)
    skin_ctr = skin_ratio_rgb(arr[y0:y1, x0:x1, :])
    scores = {"skin_ratio": skin, "skin_lo": skin_lo, "skin_center": skin_ctr}
    name = path.name.lower()
    if any(k in name for k in ("typing", "hand", "hands", "person", "people", "face", "portrait", "selfie")):
        return False, scores
    if skin > max_skin:
        return False, scores
    if skin_lo > max_skin * 1.15:
        return False, scores
    if skin_ctr > max_skin * 1.25:
        return False, scores
    return True, scores


def gate_bright_office(frames: list) -> tuple[bool, str, dict[str, float]]:
    """LTX insert gate: reject high-key bright offices."""
    import numpy as np
    if not frames:
        return False, "tech:no_frame", {"mean_luma": 0.0}
    lumas = [mean_luma_rgb(f) for f in frames]
    mean_l = float(np.mean(lumas))
    scores = {"mean_luma": mean_l, "luma": mean_l}
    if mean_l > float(MAX_DARK_OFFICE_MEAN_LUMA):
        return False, "palette:bright_office", scores
    return True, "", scores


def gate_people_hands(frames: list, *, max_skin: float | None = None) -> tuple[bool, str, dict[str, float]]:
    """LTX insert gate: reject skin/hands/face."""
    import numpy as np
    max_skin = float(MAX_SKIN_RATIO_LTX if max_skin is None else max_skin)
    if not frames:
        return False, "tech:no_frame", {"skin_ratio": 1.0}
    skins = [skin_ratio_rgb(f) for f in frames]
    smax = float(max(skins) if skins else 0.0)
    scores = {"skin_ratio": smax}
    if smax > max_skin:
        return False, "people:hands_or_face", scores
    return True, "", scores



NEGATIVE_CONCEPTS = (
    "hands", "fingers", "face", "chin", "person", "people",
    "hoodie", "back of head", "human silhouette", "hand on mouse",
    "esports event", "gaming crowd", "audience",
    "garden", "gardening tools", "frogs", "plants in soil",
    "framed photo of people", "portrait on wall",
    "bright", "white", "daylight", "sun", "stock lifestyle desk",
)
# Formal allowlist (CLIP ViT-B/32) — dark desk/office tech only
ALLOWLIST_CONCEPTS = (
    "desk",
    "technology",
    "computer monitor",
    "keyboard",
    "workspace",
    "dark office",
)
YOLO_PERSON_CLASSES = ("person", "hand", "arm", "face")
YOLO_PERSON_CONF = 0.15  # person/hand/arm/face (gold v3); other classes 0.25 if applicable
YOLO_PERSON_FACE_CONF = 0.15  # person/hand/arm/face gold v3
YOLO_STRICT_CLASSES = ("person", "hand", "arm", "face")  # all @ YOLO_PERSON_FACE_CONF 0.15
CLIP_ALLOWLIST_MIN = 0.22
MAX_OCR_TEXT_AREA_RATIO = 0.05  # reject if text bboxes cover >5% of frame
OCR_CONF_MIN = 0.60  # RapidOCR conf >60 equivalent
NEGATIVE_MAX_AVG_LUMA = 100.0
CLIP_NEG_THRESHOLD = 0.15
CLIP_PERSON_THRESHOLD = 0.12
CLIP_THEME_POS_MIN = 0.14
YUNET_FACE_SCORE_MIN = 0.82
PERSON_BLOB_MAX = 0.28


def _clip_negative_scores(path: Path) -> dict[str, float] | None:
    """Optional CLIP zero-shot vs NEGATIVE_CONCEPTS. Returns None if CLIP unavailable."""
    try:
        import torch
        from PIL import Image as _PILImage
    except Exception:
        return None
    # Try open_clip / clip / transformers lazily
    model = None
    preprocess = None
    tokenize = None
    device = "cpu"
    try:
        import open_clip  # type: ignore
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenize = open_clip.tokenize
        model = model.to(device).eval()
    except Exception:
        try:
            import clip as openai_clip  # type: ignore
            model, preprocess = openai_clip.load("ViT-B/32", device=device)
            tokenize = openai_clip.tokenize
        except Exception:
            return None
    try:
        img = preprocess(_PILImage.open(path).convert("RGB")).unsqueeze(0)
        texts = list(NEGATIVE_CONCEPTS) + ["dark office desk", "dark tech workspace", "no people"]
        with torch.no_grad():
            if hasattr(model, "encode_image"):
                image_features = model.encode_image(img)
                text_tokens = tokenize(texts)
                text_features = model.encode_text(text_tokens)
            else:
                return None
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0).cpu().tolist()
        scores = {t: float(s) for t, s in zip(texts, sims)}
        return scores
    except Exception:
        return None




_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZE = None
_CLIP_DEVICE = "cpu"


def _get_yolo():
    """Lazy-load ultralytics YOLOv8n for person/hand/arm/face."""
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL
    from ultralytics import YOLO  # type: ignore
    # downloads yolov8n.pt on first use (~6MB)
    _YOLO_MODEL = YOLO("yolov8n.pt")
    return _YOLO_MODEL


def yolo_person_scores(path: Path) -> dict[str, float]:
    """Run YOLOv8n; yolo_hit=1 if person/hand/arm/face conf>=0.15 (gold v3)."""
    scores: dict[str, float] = {
        "yolo_available": 0.0,
        "yolo_hit": 0.0,
        "yolo_max_conf": 0.0,
        "yolo_person_conf": 0.0,
        "yolo_hand_conf": 0.0,
        "yolo_arm_conf": 0.0,
        "yolo_face_conf": 0.0,
        "person_detector": 0.0,
        "yolo_thr_used": 0.0,
    }
    try:
        model = _get_yolo()
        scores["yolo_available"] = 1.0
        # predict floor at 0.15 so person/hand/arm/face thr can fire
        results = model.predict(str(path), verbose=False, conf=0.15, device="cpu")
        if not results:
            return scores
        r0 = results[0]
        names = r0.names or {}
        best_any = 0.0
        hit_name = ""
        hit_conf = 0.0
        hit_thr = 0.0
        if r0.boxes is None or len(r0.boxes) == 0:
            return scores
        for box in r0.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            cname = str(names.get(cls_id, str(cls_id))).lower()
            if cname in YOLO_PERSON_CLASSES:
                key = f"yolo_{cname}_conf"
                if key in scores:
                    scores[key] = max(float(scores[key]), conf)
                else:
                    scores[key] = conf
                thr = float(YOLO_PERSON_FACE_CONF) if cname in YOLO_STRICT_CLASSES else float(YOLO_PERSON_CONF)
                if conf >= thr and conf >= best_any:
                    best_any = conf
                    hit_name = cname
                    hit_conf = conf
                    hit_thr = thr
                elif conf > best_any and not hit_name:
                    best_any = conf
        scores["yolo_max_conf"] = float(best_any)
        scores["person_detector"] = float(best_any)
        scores["yolo_hit_class"] = 1.0 if hit_name else 0.0
        scores["yolo_thr_used"] = float(hit_thr)
        if hit_name:
            scores[f"yolo_top_{hit_name}"] = float(hit_conf)
            scores["yolo_hit"] = 1.0
            scores["yolo_hit_name_ord"] = float(sum(ord(c) for c in hit_name) % 1000)
            scores["_yolo_hit_label"] = hit_name  # type: ignore[assignment]
        return scores
    except Exception as exc:
        scores["yolo_error"] = 1.0
        scores["yolo_err_len"] = float(len(str(exc)))
        return scores


def passes_person_gate(path: Path) -> tuple[bool, str, dict[str, float]]:
    """YOLO person/hand/arm/face gate @0.15 (gold v3); other classes 0.25 if applicable."""
    ysc = yolo_person_scores(path)
    scores: dict[str, float] = {}
    hit_label = ""
    for k, v in list(ysc.items()):
        if k == "_yolo_hit_label":
            hit_label = str(v)
            continue
        try:
            scores[k] = float(v)
        except Exception:
            pass
    if float(ysc.get("yolo_hit", 0.0) or 0.0) >= 1.0:
        label = hit_label or "person"
        conf = float(ysc.get("yolo_max_conf", 0.0) or 0.0)
        return False, f"person:yolo:{label}:{conf:.3f}", scores
    return True, "", scores


def _get_clip():
    """Lazy-load open_clip / openai clip ViT-B/32 on CPU."""
    global _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZE
    if _CLIP_MODEL is not None:
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZE
    import torch
    try:
        import open_clip  # type: ignore
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        tokenize = open_clip.tokenize
        model = model.to(_CLIP_DEVICE).eval()
        _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZE = model, preprocess, tokenize
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZE
    except Exception:
        import clip as openai_clip  # type: ignore
        model, preprocess = openai_clip.load("ViT-B/32", device=_CLIP_DEVICE)
        _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZE = model, preprocess, openai_clip.tokenize
        return _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZE


def clip_allowlist_scores(path: Path) -> dict[str, float] | None:
    """CLIP ViT-B/32 zero-shot vs ALLOWLIST_CONCEPTS. None if CLIP unavailable."""
    try:
        import torch
        from PIL import Image as _PILImage
        model, preprocess, tokenize = _get_clip()
        img = preprocess(_PILImage.open(path).convert("RGB")).unsqueeze(0)
        texts = list(ALLOWLIST_CONCEPTS)
        with torch.no_grad():
            image_features = model.encode_image(img)
            text_features = model.encode_text(tokenize(texts))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0).cpu().tolist()
        out = {f"clip_{t}": float(s) for t, s in zip(texts, sims)}
        out["clip_allowlist_max"] = float(max(sims) if sims else 0.0)
        out["clip_available"] = 1.0
        return out
    except Exception:
        return None


def passes_theme_allowlist(
    path: Path,
    *,
    asset_id: str = "",
    tags: list[str] | None = None,
    primary_tag: str = "",
) -> tuple[bool, str, dict[str, float]]:
    """CLIP allowlist gate: max sim to ALLOWLIST_CONCEPTS must be >= CLIP_ALLOWLIST_MIN."""
    scores: dict[str, float] = {}
    blob = " ".join(
        [str(asset_id or ""), path.name, primary_tag, " ".join(tags or [])]
    ).lower()
    # Hard filename/primary bans (garden/esports) even before CLIP
    ban_kw = (
        "garden", "frog", "scoop", "esports", "crowd", "arena", "rog", "venue",
        "selfie", "portrait", "people", "person", "hands", "typing",
        "monstera", "succulent", "fern", "plant_leaf", "cactus", "soil", "shovel",
        "plant",
    )
    pt = (primary_tag or "").lower()
    if pt in ("plant", "garden", "frogs", "esports", "crowd"):
        return False, f"theme:ban_primary:{pt}", scores
    for kw in ban_kw:
        # only ban if kw is a path/id token (avoid 'person' substring false positives in 'personal' etc.)
        if kw in blob.split("_") or kw in blob.replace("-", " ").split() or f"_{kw}_" in f"_{blob}_":
            if kw in ("plant", "garden", "frog", "esports", "crowd", "arena", "rog", "venue",
                      "monstera", "succulent", "fern", "plant_leaf", "cactus", "soil", "shovel", "scoop"):
                scores["theme_keyword_ban"] = 1.0
                return False, f"theme:ban_kw:{kw}", scores

    clip = clip_allowlist_scores(path)
    if clip is None:
        scores["clip_available"] = 0.0
        # Without CLIP: require tech keywords (dark desk/office)
        tech_kw = ("desk", "keyboard", "monitor", "server", "pcb", "cable", "laptop", "code", "coffee", "notebook", "rack", "workspace")
        if not any(k in blob for k in tech_kw):
            return False, "theme:no_clip_no_tech_kw", scores
        scores["clip_allowlist_max"] = 0.0
        return True, "", scores

    scores.update(clip)
    mx = float(scores.get("clip_allowlist_max", 0.0))
    if mx < float(CLIP_ALLOWLIST_MIN):
        return False, f"theme:allowlist_low:{mx:.3f}", scores
    return True, "", scores


def passes_negative_gate(
    path: Path,
    *,
    asset_id: str = "",
    tags: list[str] | None = None,
    use_clip: bool = True,
) -> tuple[bool, str, dict[str, float]]:
    """Formal triple gate: dark luma + YOLO person + CLIP allowlist.

    1) avg_luma > 100 => reject
    2) YOLOv8n person/hand/arm/face conf>=0.15 => reject (gold v3)
    3) CLIP allowlist max < 0.22 => reject (kills garden/esports)
    """
    scores: dict[str, float] = {}
    try:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im).astype(np.float32)
    except Exception as exc:
        return False, f"neg:unreadable:{exc}", {"error": 1.0}

    avg_luma = mean_luma_rgb(arr)
    scores["avg_luma"] = float(avg_luma)
    scores["mean_luma"] = float(avg_luma)
    if avg_luma > float(NEGATIVE_MAX_AVG_LUMA):
        return False, f"neg:avg_luma>{NEGATIVE_MAX_AVG_LUMA:.0f}", scores

    # Keyword hard-reject on obvious person/venue filenames
    blob = " ".join([str(asset_id or ""), path.name, " ".join(tags or [])]).lower()
    for kw in ("hands", "fingers", "face", "person", "people", "typing", "hoodie", "esports", "crowd", "selfie", "portrait"):
        if kw in blob.replace("-", "_").split("_") or f"_{kw}_" in f"_{blob.replace('-', '_')}_":
            return False, f"neg:keyword:{kw}", scores

    # YOLO person gate (person/hand/arm/face @0.15 gold v3)
    pok, preason, psc = passes_person_gate(path)
    scores.update(psc)
    if not pok:
        # normalize reason prefix to neg:yolo:...
        r = preason.replace("person:yolo:", "neg:yolo:") if preason.startswith("person:yolo:") else f"neg:{preason}"
        return False, r, scores
    if float(psc.get("yolo_available", 0.0) or 0.0) < 1.0:
        scores["yolo_missing"] = 1.0

    # CLIP allowlist (also acts as theme gate)
    if use_clip:
        tok, treason, tsc = passes_theme_allowlist(
            path, asset_id=asset_id, tags=tags, primary_tag=""
        )
        for k, v in tsc.items():
            try:
                scores[k] = float(v)
            except Exception:
                pass
        if not tok:
            return False, treason if treason.startswith("theme:") else f"neg:{treason}", scores
        scores["clip_available"] = float(tsc.get("clip_available", scores.get("clip_available", 0.0)))
    else:
        scores["clip_available"] = 0.0

    return True, "", scores


def gate_motion(video: Path) -> tuple[bool, str, dict[str, float]]:
    """Mean abs frame diff via ffmpeg fps=6 sample."""
    tmp = video.with_suffix(".mot.npz")
    # extract small grayscale sequence
    pat = video.parent / f"{video.stem}_mot_%03d.jpg"
    subprocess.check_call(
        [
            _ffmpeg(), "-y", "-i", str(video), "-vf", "fps=6,scale=160:90,format=gray",
            "-frames:v", "24", str(pat),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    files = sorted(video.parent.glob(f"{video.stem}_mot_*.jpg"))
    if len(files) < 3:
        return False, "motion:too_few_frames", {"motion": 0.0}
    arrs = [np.asarray(Image.open(p).convert("L"), dtype=np.float32) for p in files]
    diffs = [np.mean(np.abs(arrs[i] - arrs[i - 1])) for i in range(1, len(arrs))]
    motion = float(np.mean(diffs)) if diffs else 0.0
    for p in files:
        try:
            p.unlink()
        except Exception:
            pass
    scores = {"motion": motion}
    # threshold: nearly static LTX; 1.8 allows slow cinematic pans
    if motion < 1.8:
        return False, "motion:almost_static", scores
    return True, "", scores


def gate_uncanny_face(frames: list[np.ndarray]) -> tuple[bool, str, dict[str, float]]:
    """Heuristic: green/wax skin blobs + left-right symmetry (twins exhibit)."""
    green_scores = []
    sym_scores = []
    skin_scores = []
    for f in frames:
        rgb = f
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        # non-human green skin
        green = (g > 90) & (g > r * 1.15) & (g > b * 1.1) & (g - np.maximum(r, b) > 25)
        green_ratio = float(green.mean())
        green_scores.append(green_ratio)
        # skin-ish
        skin = (r > 95) & (g > 40) & (b > 20) & (r > g) & (r - b > 15)
        skin_ratio = float(skin.mean())
        skin_scores.append(skin_ratio)
        # symmetry (twins / mirrored faces)
        left = rgb[:, : rgb.shape[1] // 2]
        right = np.flip(rgb[:, rgb.shape[1] // 2 :], axis=1)
        h = min(left.shape[1], right.shape[1])
        sym = 1.0 - float(np.mean(np.abs(left[:, :h] - right[:, :h])) / 255.0)
        sym_scores.append(sym)
    gmax = float(max(green_scores) if green_scores else 0)
    smax = float(max(skin_scores) if skin_scores else 0)
    sym = float(np.mean(sym_scores) if sym_scores else 0)
    scores = {"green_skin": gmax, "skin_ratio": smax, "symmetry": sym}
    if gmax > 0.04:
        return False, "horror:green_nonhuman_skin", scores
    # high skin + extreme symmetry => twin exhibit / mask pair
    if smax > 0.12 and sym > 0.82:
        return False, "uncanny:symmetric_faces_twins", scores
    # large skin blob often means big face close-up - reject for daily channel policy
    if smax > 0.22:
        return False, "uncanny:large_face_region", scores
    return True, "", scores


_OCR = None


def _get_ocr():
    global _OCR
    if _OCR is False:
        return None
    if _OCR is not None:
        return _OCR
    try:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
        return _OCR
    except Exception:
        _OCR = False
        return None


_GIBBER = re.compile(r"[A-Za-z]{3,}")
_VOWELS = set("aeiouyAEIOUY")


def _looks_gibberish(word: str) -> bool:
    w = re.sub(r"[^A-Za-z]", "", word)
    if len(w) < 4:
        return False
    # mixed case soup / almost-words
    if re.search(r"[A-Z]{3,}[a-z]{2,}[A-Z]", word):
        return True
    vowels = sum(1 for c in w if c in _VOWELS)
    if vowels / len(w) < 0.15:
        return True
    # high consonant clusters
    if re.search(r"[bcdfghjklmnpqrstvwxz]{5,}", w, re.I):
        return True
    # known junk from Ou5y
    junk = ("philk", "liarly", "svn", "wetiro", "2aib", "irooo")
    low = w.lower()
    if any(j in low for j in junk):
        return True
    return False



def _bbox_area(box) -> float:
    """Area of RapidOCR 4-point box (or xyxy)."""
    try:
        import numpy as _np
        pts = _np.asarray(box, dtype=_np.float64).reshape(-1, 2)
        if pts.shape[0] < 2:
            return 0.0
        if pts.shape[0] == 2:
            return float(abs(pts[1, 0] - pts[0, 0]) * abs(pts[1, 1] - pts[0, 1]))
        # shoelace
        x, y = pts[:, 0], pts[:, 1]
        return float(abs(_np.dot(x, _np.roll(y, -1)) - _np.dot(y, _np.roll(x, -1))) * 0.5)
    except Exception:
        return 0.0


def passes_ocr_gate(
    path: Path,
    *,
    max_ratio: float | None = None,
    conf_min: float | None = None,
    resize_wh: tuple[int, int] = (800, 600),
) -> tuple[bool, str, dict[str, float]]:
    """Reject frames whose OCR text boxes cover > max_ratio of the frame.

    Resize to ~800x600 for stable detection. Uses RapidOCR when available.
    conf>60 equivalent via conf_min (default 0.60).
    """
    max_ratio = float(MAX_OCR_TEXT_AREA_RATIO if max_ratio is None else max_ratio)
    conf_min = float(OCR_CONF_MIN if conf_min is None else conf_min)
    scores: dict[str, float] = {
        "ocr_text_area_ratio": 0.0,
        "ocr_box_count": 0.0,
        "ocr_frame_area": 0.0,
        "ocr_text_area": 0.0,
    }
    ocr = _get_ocr()
    if ocr is None:
        scores["ocr_available"] = 0.0
        return True, "", scores
    scores["ocr_available"] = 1.0
    tmp_path = None
    try:
        im = Image.open(path).convert("RGB")
        w0, h0 = im.size
        target_w, target_h = int(resize_wh[0]), int(resize_wh[1])
        im_r = im.resize((target_w, target_h))
        import tempfile
        fd, tmp_s = tempfile.mkstemp(suffix=".jpg")
        import os
        os.close(fd)
        tmp_path = Path(tmp_s)
        im_r.save(tmp_path, quality=92)
        frame_area = float(target_w * target_h)
        scores["ocr_frame_area"] = frame_area
        result, _ = ocr(str(tmp_path))
        text_area = 0.0
        n_boxes = 0
        if result:
            for line in result:
                if not line or len(line) < 2:
                    continue
                box = line[0]
                conf = 1.0
                if len(line) >= 3:
                    try:
                        conf = float(line[2])
                    except Exception:
                        conf = 1.0
                # RapidOCR conf often 0-1; sometimes 0-100
                conf_n = conf / 100.0 if conf > 1.5 else conf
                if conf_n < conf_min:
                    continue
                a = _bbox_area(box)
                if a > 0:
                    text_area += a
                    n_boxes += 1
        ratio = float(text_area / frame_area) if frame_area > 0 else 0.0
        scores["ocr_text_area"] = float(text_area)
        scores["ocr_box_count"] = float(n_boxes)
        scores["ocr_text_area_ratio"] = float(ratio)
        if ratio > max_ratio:
            return False, f"ocr:text_area_ratio:{ratio:.4f}>{max_ratio:.2f}", scores
        return True, "", scores
    except Exception as exc:
        scores["ocr_error"] = 1.0
        scores["ocr_err_len"] = float(len(str(exc)))
        return True, "", scores  # soft-open on OCR crash
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink()
            except Exception:
                pass


def gate_ocr(frames_paths: list[Path]) -> tuple[bool, str, dict[str, float]]:
    ocr = _get_ocr()
    scores = {"ocr_chars": 0.0, "ocr_gibber_hits": 0.0}
    if ocr is None:
        # without OCR: soft fail open (do not block pipeline install)
        scores["ocr_available"] = 0.0
        return True, "", scores
    scores["ocr_available"] = 1.0
    texts: list[str] = []
    gibber = 0
    for p in frames_paths:
        try:
            result, _ = ocr(str(p))
        except Exception:
            continue
        if not result:
            continue
        for line in result:
            # RapidOCR: [box, text, conf]
            if len(line) < 2:
                continue
            t = str(line[1] if not isinstance(line[1], (list, tuple)) else line[1][0])
            texts.append(t)
            for w in _GIBBER.findall(t):
                if _looks_gibberish(w):
                    gibber += 1
    joined = " ".join(texts)
    scores["ocr_chars"] = float(len(joined))
    scores["ocr_gibber_hits"] = float(gibber)
    # STRICT: any detectable text / mark-like OCR on LTX screens -> reject -> pool still
    letters = re.findall(r"[A-Za-z0-9]", joined)
    scores["ocr_alnum"] = float(len(letters))
    if gibber >= 1:
        return False, "ocr:gibberish_ui_text", scores
    if len(joined.strip()) >= 3:  # any OCR string of 3+ chars
        return False, "ocr:any_detectable_text", scores
    if len(re.findall(r"[A-Za-z]{2,}", joined)) >= 1:
        return False, "ocr:readable_screen_text", scores
    if len(letters) >= 2:
        return False, "ocr:mark_like_glyphs", scores
    return True, "", scores


def gate_style(prev_hist: np.ndarray | None, frame: np.ndarray) -> tuple[bool, str, dict[str, float], np.ndarray]:
    hist = np.concatenate([
        np.histogram(frame[:, :, c], bins=32, range=(0, 255))[0].astype(np.float32)
        for c in range(3)
    ])
    hist = hist / (hist.sum() + 1e-6)
    if prev_hist is None:
        return True, "", {"style_corr": 1.0}, hist
    corr = float(np.corrcoef(prev_hist, hist)[0, 1])
    scores = {"style_corr": corr}
    if corr < 0.35:
        return False, "style:palette_jump", scores, hist
    return True, "", scores, hist


def gate_clip_proxy(brief: str, ocr_ok: bool, uncanny_ok: bool) -> tuple[bool, str, dict[str, float]]:
    """Lightweight stand-in until full CLIP embeds are wired.

    Reject off-topic stills if brief asks for desk/agent/pipeline but we already
    failed face/OCR (handled elsewhere). Keyword: rooftop/cottage/aerial city
    briefs without agent nouns get soft pass here; caller passes script_beat.
    """
    beat = (brief or "").lower()
    scores = {"clip_proxy": 1.0}
    off = ("cottage", "rooftop", "suburb", "aerial estate", "residential roofs")
    on = ("agent", "desk", "laptop", "server", "pipeline", "schematic", "terminal", "queue", "keyboard", "monitor")
    if any(x in beat for x in off) and not any(x in beat for x in on):
        scores["clip_proxy"] = 0.1
        return False, "clip:off_topic_broll", scores
    return True, "", scores


def evaluate_clip(
    video: Path,
    *,
    script_beat: str,
    visual_brief: str,
    work_dir: Path,
    prev_hist: np.ndarray | None = None,
) -> GateResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    scores: dict[str, float] = {}
    kfs = extract_keyframes(video, work_dir / "kf")
    if not kfs:
        return GateResult(False, ["tech:no_keyframe"], {}, None)
    frames = [_load_rgb(p) for p in kfs]
    key = str(kfs[len(kfs) // 2])

    ok, reason, sc = gate_tech(frames)
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    ok, reason, sc = gate_motion(video)
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    ok, reason, sc = gate_uncanny_face(frames)
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    ok, reason, sc = gate_people_hands(frames)
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    ok, reason, sc = gate_bright_office(frames)
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    # OCR: area-ratio <=5% (primary) + gibberish-only from gate_ocr
    mid_kf = Path(key) if key else (kfs[len(kfs) // 2] if kfs else None)
    if mid_kf is not None:
        ok, reason, sc = passes_ocr_gate(mid_kf)
        scores.update(sc)
        if not ok:
            return GateResult(False, [reason], scores, key)
    ok, reason, sc = gate_ocr(kfs)
    scores.update(sc)
    if (not ok) and ("gibber" in str(reason).lower() or "scribbl" in str(reason).lower()):
        return GateResult(False, [reason], scores, key)
    # ignore gate_ocr any_detectable_text here — area gate is the PASS criterion

    # Formal negative gate on mid keyframe (YOLO person/hand/arm/face@0.15 / CLIP / luma)
    if mid_kf is not None:
        ok, reason, sc = passes_negative_gate(
            mid_kf,
            asset_id=f"ltx_desk_{video.stem}",
            tags=["desk", "workspace", "keyboard", "monitor", "dark office"],
        )
        scores.update(sc)
        if not ok:
            return GateResult(False, [reason], scores, key)

    ok, reason, sc, hist = gate_style(prev_hist, frames[len(frames) // 2])
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    brief = f"{script_beat} {visual_brief}"
    ok, reason, sc = gate_clip_proxy(brief, True, True)
    scores.update(sc)
    if not ok:
        return GateResult(False, [reason], scores, key)

    # stash hist on result via scores side channel path file
    hist_path = work_dir / "last_hist.npy"
    np.save(hist_path, hist)
    scores["hist_path"] = 1.0
    return GateResult(True, [], scores, key)



def is_geometric_still(path: Path) -> tuple[bool, dict[str, float]]:
    """Reject flat rect / grid / node-diagram pool candidates.

    Soft dark-desk photos can be low-contrast; only flag *diagram-like* stubs:
    near-solid fills, regular LED/dot grids, or node-network line art.
    """
    try:
        im = Image.open(path).convert("RGB").resize((320, 180))
        arr = np.asarray(im).astype(np.float32)
    except Exception:
        return True, {"var": 0.0, "flat_ratio": 1.0, "edge": 0.0, "grid": 0.0}
    var = float(arr.var())
    gray = arr.mean(axis=2)
    bh = bw = 16
    flats = blocks = 0
    for y in range(0, gray.shape[0] - bh, bh):
        for x in range(0, gray.shape[1] - bw, bw):
            blocks += 1
            if float(gray[y : y + bh, x : x + bw].std()) < 6.0:
                flats += 1
    flat_ratio = flats / max(blocks, 1)
    gx = float(np.abs(np.diff(gray, axis=1)).mean())
    gy = float(np.abs(np.diff(gray, axis=0)).mean())
    edge = (gx + gy) / 2.0

    # Dot-grid / LED board: many small bright peaks on dark field
    thr = float(np.percentile(gray, 92))
    peaks = (gray >= thr) & (gray > gray.mean() + 25)
    # connected-ish peak density in a coarse grid
    ph, pw = 10, 10
    occupied = 0
    cells = 0
    for y in range(0, gray.shape[0] - ph, ph):
        for x in range(0, gray.shape[1] - pw, pw):
            cells += 1
            if peaks[y : y + ph, x : x + pw].mean() > 0.04:
                occupied += 1
    grid_score = occupied / max(cells, 1)

    # Teal/cyan dominant flat panels (classic bad stubs)
    mean_rgb = arr.reshape(-1, 3).mean(axis=0)
    tealish = float(mean_rgb[2] > mean_rgb[0] + 8 and mean_rgb[1] > mean_rgb[0] + 5)

    # Glowing soft rectangle (m02): bright center panel vs dark border
    H, W = gray.shape
    cy0, cy1 = int(H * 0.12), int(H * 0.55)
    cx0, cx1 = int(W * 0.12), int(W * 0.78)
    center = gray[cy0:cy1, cx0:cx1]
    border = np.concatenate([
        gray[: int(H * 0.08), :].ravel(),
        gray[int(H * 0.85) :, :].ravel(),
        gray[:, : int(W * 0.06)].ravel(),
        gray[:, int(W * 0.94) :].ravel(),
    ])
    glow_delta = float(center.mean() - border.mean())

    name = path.stem.lower()

    name_hit = any(k in name for k in ("schematic", "node", "grid", "leds", "diagram", "bokeh", "desk_glow", "blank_monitor", "monitor_bokeh", "wide_bokeh", "keyboard_soft", "laptop_soft", "moody_keys", "corner_desk"))

    is_geo = bool(
        name_hit
        # m02 stubs: large glow delta AND grainy soft edges from KenBurns/noise
        or (glow_delta > 35.0 and edge > 30.0 and var < 7000)
        or (glow_delta > 45.0 and 35.0 < edge < 60.0)
        or (var < 60 and flat_ratio > 0.90)  # near-solid fill
        or (grid_score > 0.35 and var < 800)  # LED/dot boards
        or (flat_ratio > 0.95 and edge < 1.2 and var < 250)  # empty rect panels
        or (tealish > 0.5 and flat_ratio > 0.85 and var < 200)
    )
    return is_geo, {
        "var": var,
        "flat_ratio": flat_ratio,
        "edge": edge,
        "grid": float(grid_score),
        "tealish": tealish,
        "glow_delta": glow_delta,
    }


def make_fallback_still(
    dest: Path,
    *,
    beat: str,
    title: str,
    index: int,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Photoreal desk/keyboard still from photo_pool (NO abstract glowing rectangles)."""
    from photo_fallback import make_photoreal_still
    return make_photoreal_still(
        dest, beat=beat, title=title, index=index, width=width, height=height
    )


def still_to_kenburns_clip(
    still: Path,
    dest: Path,
    *,
    seconds: float = 4.0,
    fps: int = 24,
    max_still_hold_sec: float | None = None,
) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Cap hold — never stretch one still beyond MAX_STILL_HOLD_SEC
    try:
        from video_assembler import load_assembler_config
        cap = float(load_assembler_config().get("max_still_hold_sec", 5.0))
    except Exception:
        cap = 5.0
    if max_still_hold_sec is not None:
        cap = float(max_still_hold_sec)
    seconds = min(float(seconds), float(cap))
    if seconds > float(cap) + 1e-6:
        print(f"[kenburns] ASSERT soft: requested {seconds:.3f}s > cap {cap:.3f}s — clamping", flush=True)
        seconds = float(cap)
    frames = max(24, int(round(seconds * fps)))
    # Hard -t + frames:v so zoompan cannot overshoot duration
    filt = (
        f"scale=1280:720:force_original_aspect_ratio=decrease,"
        f"pad=1280:720:(ow-iw)/2:(oh-ih)/2,"
        f"zoompan=z='min(1.08,1+0.0009*on)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d={frames}:s=1280x720:fps={fps}"
    )
    subprocess.check_call(
        [
            _ffmpeg(), "-y", "-loop", "1", "-i", str(still),
            "-vf", filt, "-frames:v", str(frames),
            "-t", f"{seconds:.3f}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-an", str(dest),
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    # Probe-enforce: soft-fail log if muxed still exceeds cap
    try:
        from video_assembler import probe_duration
        got = float(probe_duration(dest))
        if got > float(cap) + 0.08:
            print(
                f"[kenburns] ASSERT FAIL soft: {dest.name} probed={got:.3f}s > max_still_hold={cap:.3f}s",
                flush=True,
            )
            # re-trim hard
            trimmed = dest.with_suffix(".trim.mp4")
            subprocess.check_call(
                [
                    _ffmpeg(), "-y", "-i", str(dest), "-t", f"{cap:.3f}",
                    "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-an", str(trimmed),
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            trimmed.replace(dest)
            print(f"[kenburns] re-trimmed -> {cap:.3f}s", flush=True)
        else:
            print(f"[kenburns] ok hold={got:.3f}s <= {cap:.3f}s ({dest.name})", flush=True)
    except Exception as exc:
        print(f"[kenburns] warning: duration assert skipped ({exc})", flush=True)
    return dest

def pick_thumbnail(accepted_keyframes: list[Path], dest: Path) -> Path | None:
    """Prefer object/desk stills: mid luma, mid variance, not face-heavy."""
    if not accepted_keyframes:
        return None
    best = None
    best_score = -1e9
    for p in accepted_keyframes:
        arr = _load_rgb(p)
        luma = float(arr.mean())
        var = float(arr.var())
        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        skin = float(((r > 95) & (g > 40) & (b > 20) & (r > g) & (r - b > 15)).mean())
        score = -abs(luma - 90) + min(var, 3000) / 100 - skin * 500
        if score > best_score:
            best_score = score
            best = p
    if best is None:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    Image.open(best).convert("RGB").save(dest, quality=92)
    return dest
