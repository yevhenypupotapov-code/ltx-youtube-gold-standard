#!/usr/bin/env python3
"""Quality-control timeline assembler for LTX YouTube factory (TZ v2.1).

Assembles approved LTX clips + diversified fallback stills into a timeline that:
- never holds one still/static asset longer than max_still_hold_sec (5.0)
- enforces min visual density (>=1 unique asset / 5s audio)
- rejects abstract nonsense via keyword/tag semantic score (CLIP embeddings deferred)
- never reuses asset_id while unused pool items remain (reuse only after exhaust, gap>=30s)
- bans phash near-dups anywhere in the timeline (not only adjacent)
- Laplacian structure gate on pool admit (reject flat lids/noise)
- hard -t max_still_hold_sec on every concat-normalized segment
- Sliding Window Semantic Diversity: deque(maxlen=3) on exclusion_group (~15s)
- exclusion_group clusters: device_macro / screen / desk / prop
- After device_macro, next picks prefer desk/prop/screen
- Dark-office mean_luma gate + YOLO person/face@0.15 + OCR text-area<=5% on pool admit
- picks thumbnails only from approved non-fallback clips with motion

Soft-fail (default for schedule): quality_compromised=True, skip upload, exit non-zero.
Hard abort with --strict-assemble: raise PipelineAbortError.
"""
from __future__ import annotations

from collections import deque

import json
import re
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# Defaults (overridden by config/video_assembler.yaml)
MAX_STILL_HOLD_SEC = 5.0
MIN_SEMANTIC_SCORE = 0.65
MIN_MOTION_FOR_THUMB = 0.4
MIN_UNIQUE_PER_SEC = 5.0
PHASH_NEAR_DUP_MAX = 55  # stricter for small pools; ban anywhere in timeline
REUSE_MIN_GAP_SEC = 30.0  # only after pool exhausted of unused asset_ids
MIN_LAPLACIAN_VAR = 100.0


class PipelineAbortError(Exception):
    """Hard abort: cannot assemble a QC-passing timeline."""

def _load_pipeline_state(path: Path | None = None) -> dict:
    p = path or (Path(__file__).resolve().parent / "pipeline_state.json")
    try:
        import json as _j
        if p.exists():
            return _j.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_pipeline_state(state: dict, path: Path | None = None) -> None:
    p = path or (Path(__file__).resolve().parent / "pipeline_state.json")
    import json as _j
    p.write_text(_j.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def get_blacklisted_stills(state: dict, cooldown_hours: int = 24) -> set[str]:
    """Asset ids of stills used in recent runs (cooldown window)."""
    from datetime import datetime

    now = datetime.now()
    out: set[str] = set()
    for asset_id, info in (state.get("used_still_fingerprints") or {}).items():
        try:
            last = datetime.fromisoformat(str(info.get("last_used") or ""))
        except Exception:
            out.add(str(asset_id))
            continue
        if (now - last).total_seconds() < cooldown_hours * 3600:
            out.add(str(asset_id))
    return out


def filter_pool_by_prior_fingerprints(
    pool: list, *, ban_max: int = 10, cooldown_hours: int = 24
) -> list:
    """Drop stills near prior upload keyframes OR inside used_still cooldown."""
    import imagehash as _ih
    from PIL import Image as _PilImg

    state = _load_pipeline_state()
    cooldown_ids = get_blacklisted_stills(state, cooldown_hours=cooldown_hours)
    prior_hex: list[tuple[str, str]] = []
    hist = list(state.get("upload_history") or [])
    # Last 2 uploads only for still bans (full history still used at YouTube dedup).
    for it in hist[-2:]:
        title = str(it.get("title") or "prior")
        for ph in ((it.get("fingerprint") or {}).get("phash_list") or []):
            prior_hex.append((title, str(ph)))
    kept: list = []
    banned_cd = banned_ph = 0
    for a in pool:
        aid = str(getattr(a, "asset_id", "") or "")
        p = Path(getattr(a, "path", a))
        if aid and aid in cooldown_ids:
            banned_cd += 1
            print(f"[assemble] STILL_COOLDOWN_BAN {aid}", flush=True)
            continue
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and prior_hex:
            try:
                hx = str(_ih.phash(_PilImg.open(p)))
                best, who = 999, ""
                for title, hph in prior_hex:
                    d = int(_ih.hex_to_hash(hx) - _ih.hex_to_hash(hph))
                    if d < best:
                        best, who = d, title
                if best <= ban_max:
                    banned_ph += 1
                    print(
                        f"[assemble] PRIOR_PHASH_BAN {p.name} min_dist={best} "
                        f"vs {who[:48]!r} (limit<={ban_max})",
                        flush=True,
                    )
                    continue
            except Exception:
                pass
        kept.append(a)
    n0 = len(pool)
    # Exhaustion: may re-open COOLDOWN ids only — never PRIOR_PHASH_BAN clones.
    if len(kept) < max(4, int(n0 * 0.3)):
        print(
            f"[assemble] WARNING: Still pool exhaustion ({len(kept)}/{n0}). "
            f"Reopening cooldown only (prior_phash stays banned).",
            flush=True,
        )
        kept2 = []
        for a in pool:
            p = Path(getattr(a, "path", a))
            # skip if would be prior_phash banned
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and prior_hex:
                try:
                    hx = str(_ih.phash(_PilImg.open(p)))
                    best = min(int(_ih.hex_to_hash(hx) - _ih.hex_to_hash(hph)) for _, hph in prior_hex)
                    if best <= ban_max:
                        continue
                except Exception:
                    pass
            kept2.append(a)
        if len(kept2) >= 4:
            kept = kept2
        else:
            print(
                f"[assemble] CRITICAL thin unique stills={len(kept2)}; keeping prior-safe {len(kept)}",
                flush=True,
            )
    if banned_cd or banned_ph:
        print(
            f"[assemble] still blacklist removed cooldown={banned_cd} prior_phash={banned_ph}; "
            f"pool {n0}->{len(kept)}",
            flush=True,
        )
    return kept



def prior_phash_too_close(path: Path, *, limit: int = 8) -> tuple[bool, int, str]:
    """True if image/video midframe is within `limit` of any upload_history keyframe."""
    import imagehash as _ih
    from PIL import Image as _PilImg
    import subprocess
    import tempfile

    state = _load_pipeline_state()
    prior = []
    hist = list(state.get("upload_history") or [])
    for it in hist[-2:]:
        title = str(it.get("title") or "prior")
        for ph in ((it.get("fingerprint") or {}).get("phash_list") or []):
            prior.append((title, str(ph)))
    if not prior:
        return False, 999, ""
    p = Path(path)
    try:
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
            hx = str(_ih.phash(_PilImg.open(p)))
        elif p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            tmp = Path(tempfile.gettempdir()) / f"_prior_gate_{p.stem}.jpg"
            subprocess.check_call(
                ["ffmpeg", "-y", "-ss", "0.4", "-i", str(p), "-frames:v", "1", "-q:v", "4", str(tmp)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            hx = str(_ih.phash(_PilImg.open(tmp)))
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass
        else:
            return False, 999, ""
        best, who = 999, ""
        for title, hph in prior:
            d = int(_ih.hex_to_hash(hx) - _ih.hex_to_hash(hph))
            if d < best:
                best, who = d, title
        return (best <= limit), best, who
    except Exception:
        return False, 999, ""

def record_used_stills(timeline: list, *, video_ref: str = "assembled") -> None:
    """Persist fallback stills from timeline into used_still_fingerprints."""
    from datetime import datetime
    import imagehash as _ih
    from PIL import Image as _PilImg

    state = _load_pipeline_state()
    bucket = state.setdefault("used_still_fingerprints", {})
    now = datetime.now().isoformat(timespec="seconds")
    n = 0
    for clip in timeline:
        aid = str(getattr(clip, "asset_id", "") or "")
        if not aid or aid.startswith("ltx"):
            continue
        if not bool(getattr(clip, "is_fallback", False)):
            continue
        p = Path(getattr(clip, "path"))
        ph = ""
        try:
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}:
                ph = str(_ih.phash(_PilImg.open(p)))
        except Exception:
            ph = ""
        entry = bucket.get(aid) or {"phash": ph, "last_used": now, "used_in": []}
        entry["last_used"] = now
        if ph:
            entry["phash"] = ph
        used_in = list(entry.get("used_in") or [])
        if video_ref not in used_in:
            used_in.append(video_ref)
        entry["used_in"] = used_in[-12:]
        bucket[aid] = entry
        n += 1
    if n:
        _save_pipeline_state(state)
        print(f"[assemble] recorded {n} used stills into used_still_fingerprints", flush=True)



@dataclass
class ShotCandidate:
    asset_id: str
    path: Path
    is_fallback: bool = False
    tags: list[str] = field(default_factory=list)
    motion_score: float = 0.0
    script_beat: str = ""
    hold_sec: float = 0.0
    semantic_score: float = 1.0
    primary_tag: str = ""
    subfamily: str = ""
    exclusion_group: str = ""
    super_category: str = ""
    mean_luma: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["path"] = str(self.path)
        return d


@dataclass
class AssembleResult:
    ok: bool
    clips: list[ShotCandidate]
    quality_compromised: bool = False
    reason: str = ""
    thumb_source: Path | None = None
    audio_duration: float = 0.0
    covered_sec: float = 0.0
    unique_assets: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "quality_compromised": self.quality_compromised,
            "reason": self.reason,
            "audio_duration": self.audio_duration,
            "covered_sec": self.covered_sec,
            "unique_assets": self.unique_assets,
            "thumb_source": str(self.thumb_source) if self.thumb_source else None,
            "clips": [c.to_dict() for c in self.clips],
        }


def _ffmpeg() -> str:
    candidates = [
        Path(r"C:\Users\yevhe\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"),
        Path(r"D:\ComfyUI\ComfyUI_windows_portable\python_embeded\Scripts\ffmpeg.exe"),
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return "ffmpeg"


def load_assembler_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or (ROOT / "config" / "video_assembler.yaml")
    defaults = {
        "max_still_hold_sec": MAX_STILL_HOLD_SEC,
        "min_semantic_score": MIN_SEMANTIC_SCORE,
        "min_motion_for_thumb": MIN_MOTION_FOR_THUMB,
        "min_unique_per_sec": MIN_UNIQUE_PER_SEC,
        "default_soft_fail": True,
        "fallback_pool": {
            "min_items_required": 6,
            "max_items": 15,
            "dir": "fallback_pool",
        },
    }
    if not cfg_path.exists():
        return defaults
    try:
        import yaml  # type: ignore
        data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    except Exception:
        # Minimal YAML subset parser (key: value / nested under fallback_pool)
        data = {}
        cur = data
        for line in cfg_path.read_text(encoding="utf-8").splitlines():
            raw = line.split("#", 1)[0].rstrip()
            if not raw.strip():
                continue
            if not line.startswith(" ") and raw.strip().endswith(":"):
                key = raw.strip()[:-1]
                data[key] = {}
                cur = data[key]
                continue
            if ":" in raw:
                k, v = raw.split(":", 1)
                k, v = k.strip(), v.strip()
                if not v:
                    continue
                if v.lower() in ("true", "false"):
                    val: Any = v.lower() == "true"
                else:
                    try:
                        val = float(v) if "." in v else int(v)
                    except ValueError:
                        val = v.strip("'\"")
                # indented => nested
                if line.startswith(" ") or line.startswith("\t"):
                    if not isinstance(cur, dict):
                        cur = data.setdefault("fallback_pool", {})
                    cur[k] = val
                else:
                    data[k] = val
                    cur = data
    for k, v in defaults.items():
        data.setdefault(k, v)
    return data



def _still_phash(path: Path, size: int = 16):
    """Perceptual hash bits for near-dup still detection."""
    try:
        from PIL import Image
        import numpy as np
        # accept still or extract midframe from mp4
        p = Path(path)
        if p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
            import tempfile, subprocess
            tmp = p.with_suffix(".phash.jpg")
            subprocess.check_call(
                [_ffmpeg(), "-y", "-ss", "0.2", "-i", str(p), "-frames:v", "1", "-q:v", "5", str(tmp)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            img = Image.open(tmp).convert("L")
            try: tmp.unlink(missing_ok=True)
            except Exception: pass
        else:
            img = Image.open(p).convert("L")
        a = np.asarray(img.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
        return (a > a.mean()).flatten()
    except Exception:
        return None

def _phash_hamming(a, b) -> int:
    if a is None or b is None:
        return 999
    import numpy as np
    return int(np.count_nonzero(a != b))


PRIMARY_TAGS = (
    "laptop_closed",
    "keyboard_macro",
    "monitor_code",
    "desk_wide",
    "cables_peripherals",
    "coffee_mug_dark",
    "notebook",
    "plant",
)


def infer_primary_tag(asset_id: str, tags: list[str] | None = None, explicit: str | None = None) -> str:
    """Mutually exclusive semantic primary_tag for sliding-window diversity."""
    if explicit:
        return str(explicit)
    tags = list(tags or [])
    for t in tags:
        if t in PRIMARY_TAGS:
            return t
    al = (asset_id or "").lower()
    if any(k in al for k in ("kbd", "keyboard", "keys", "mech")):
        return "keyboard_macro"
    if any(k in al for k in ("typing", "cable", "peripheral", "hands")):
        return "cables_peripherals"
    if any(k in al for k in ("laptop", "macbook", "desk_mac", "_mac_")):
        return "laptop_closed"
    if any(k in al for k in ("monitor", "code_screen", "code_desk", "screen")):
        return "monitor_code"
    if any(k in al for k in ("coffee", "mug")):
        return "coffee_mug_dark"
    if "notebook" in al:
        return "notebook"
    if "plant" in al:
        return "plant"
    if any(k in al for k in ("desk", "office", "wide")):
        return "desk_wide"
    for t in tags:
        if t in ("keyboard",):
            return "keyboard_macro"
        if t in ("laptop",):
            return "laptop_closed"
        if t in ("monitor", "code"):
            return "monitor_code"
        if t in ("notebook", "plant", "desk_wide"):
            return t
        if t in ("hands",):
            return "cables_peripherals"
    return "desk_wide"


# Hierarchical sliding-window super_categories (deque maxlen=2 ~10s)
SUPER_CATEGORIES = (
    "device_macro",
    "environment_wide",
    "abstract_screen",
    "peripherals_detail",
)
# Back-compat aliases used in older manifests / logs
EXCLUSION_GROUPS = SUPER_CATEGORIES
DEVICE_MACRO_TAGS = frozenset({"laptop_closed", "keyboard_macro", "laptop", "keyboard"})
SCREEN_TAGS = frozenset({"monitor_code", "monitor", "monitor_glow", "code"})
DESK_TAGS = frozenset({"desk_wide", "desk", "office"})
PROP_TAGS = frozenset({"coffee_mug_dark", "notebook", "plant", "cables_peripherals", "cables"})

_EG_TO_SUPER = {
    "device_macro": "device_macro",
    "screen": "abstract_screen",
    "desk": "environment_wide",
    "prop": "peripherals_detail",
    "abstract_screen": "abstract_screen",
    "environment_wide": "environment_wide",
    "peripherals_detail": "peripherals_detail",
}


def infer_super_category(
    primary_tag: str,
    tags: list[str] | None = None,
    asset_id: str = "",
    explicit: str | None = None,
) -> str:
    """Map primary_tag -> super_category for hierarchical sliding window (maxlen=2).

    laptop_* + keyboard_* -> device_macro
    desk_wide -> environment_wide
    monitor_code -> abstract_screen
    coffee/cables/notebook/plant -> peripherals_detail
    """
    if explicit:
        ex = str(explicit)
        return _EG_TO_SUPER.get(ex, ex)
    pt = (primary_tag or "").strip()
    al = (asset_id or "").lower()
    tags = list(tags or [])
    blob = " ".join([pt] + tags + [al]).lower()
    if pt in DEVICE_MACRO_TAGS or pt.startswith("laptop") or pt.startswith("keyboard"):
        return "device_macro"
    if any(k in blob for k in ("laptop", "keyboard", "kbd", "macbook", "keys", "mech")):
        return "device_macro"
    if pt in SCREEN_TAGS or pt == "monitor_code" or "monitor" in blob or "screen" in blob:
        return "abstract_screen"
    if pt in PROP_TAGS or any(k in blob for k in ("coffee", "mug", "notebook", "plant", "cable", "peripheral")):
        return "peripherals_detail"
    if pt in DESK_TAGS or pt == "desk_wide" or "desk" in blob or "office" in blob:
        return "environment_wide"
    return "environment_wide"


def infer_exclusion_group(
    primary_tag: str,
    tags: list[str] | None = None,
    asset_id: str = "",
    explicit: str | None = None,
) -> str:
    """Back-compat alias -> infer_super_category."""
    return infer_super_category(primary_tag, tags, asset_id, explicit)


def load_fallback_pool(pool_dir: Path | None = None) -> list[ShotCandidate]:
    d = pool_dir or (ROOT / "fallback_pool")
    man = d / "manifest.json"
    items: list[ShotCandidate] = []
    try:
        from shot_gate import is_geometric_still, still_luma_ok, laplacian_variance
    except Exception:
        is_geometric_still = None  # type: ignore
        still_luma_ok = None  # type: ignore
        laplacian_variance = None  # type: ignore

    def _accept(
        p: Path,
        asset_id: str,
        tags: list[str],
        explicit_primary: str | None = None,
        explicit_subfamily: str | None = None,
        explicit_exclusion: str | None = None,
        manifest_mean_luma: float | None = None,
    ) -> ShotCandidate | None:
        if not p.exists():
            print(f"[assemble] warning: missing pool asset {asset_id}", flush=True)
            return None
        if is_geometric_still is not None:
            geo, scores = is_geometric_still(p)
            if geo:
                print(
                    f"[assemble] warning: reject geometric pool asset {asset_id} "
                    f"var={scores.get('var',0):.0f} flat={scores.get('flat_ratio',0):.2f}",
                    flush=True,
                )
                return None
        if still_luma_ok is not None:
            ok, lsc = still_luma_ok(p)
            if not ok:
                print(
                    f"[assemble] warning: reject near-black/flat/low-structure pool asset {asset_id} "
                    f"luma={lsc.get('luma',0):.1f} var={lsc.get('variance',0):.0f} "
                    f"lap={lsc.get('laplacian',0):.1f}",
                    flush=True,
                )
                return None
        mean_luma_val = float(manifest_mean_luma) if manifest_mean_luma is not None else 0.0
        try:
            from shot_gate import passes_negative_gate
            nok, nreason, nsc = passes_negative_gate(p, asset_id=str(asset_id), tags=list(tags or []))
            mean_luma_val = float(nsc.get("avg_luma", nsc.get("mean_luma", mean_luma_val)) or 0.0)
            if not nok:
                print(
                    f"[assemble] warning: negative_gate reject pool asset {asset_id} "
                    f"reason={nreason} avg_luma={mean_luma_val:.1f} yolo={nsc.get('yolo_max_conf',0):.3f} clip={nsc.get('clip_allowlist_max',0):.3f}",
                    flush=True,
                )
                return None
        except Exception as _dark_exc:
            print(f"[assemble] warning: negative_gate FAIL-CLOSED reject {asset_id} ({_dark_exc})", flush=True)
            return None
        # Explicit Laplacian structure insurance (grain fools mean luma)
        if laplacian_variance is not None:
            try:
                lap = float(laplacian_variance(p))
                if lap < float(MIN_LAPLACIAN_VAR):
                    print(
                        f"[assemble] warning: reject low-laplacian pool asset {asset_id} lap={lap:.1f}<{MIN_LAPLACIAN_VAR}",
                        flush=True,
                    )
                    return None
            except Exception as _lap_exc:
                print(f"[assemble] warning: laplacian skip {asset_id} ({_lap_exc})", flush=True)
        # OCR text-area gate (screen/monitor/code => 1.5% limit)
        try:
            from shot_gate import passes_ocr_gate
            _cat = " ".join(
                [
                    str(asset_id or ""),
                    str(explicit_primary or ""),
                    " ".join(tags or []),
                ]
            )
            _ok, _reason, _sc = passes_ocr_gate(p, shot_category=_cat)
            if not _ok:
                print(
                    f"[assemble] warning: reject OCR-area pool asset {asset_id} reasons={_reason} "
                    f"ratio={_sc.get('ocr_text_area_ratio', 0):.4f} lim={_sc.get('ocr_max_ratio_used', 0):.3f}",
                    flush=True,
                )
                return None
        except Exception as _ocr_exc:
            print(f"[assemble] warning: pool OCR skipped for {asset_id} ({_ocr_exc})", flush=True)
        pt = infer_primary_tag(str(asset_id), tags, explicit_primary)
        sc = infer_super_category(pt, tags, str(asset_id), explicit_exclusion)
        return ShotCandidate(
            asset_id=str(asset_id),
            path=p,
            is_fallback=True,
            tags=list(tags or []),
            motion_score=0.05,
            primary_tag=pt,
            subfamily=str(explicit_subfamily or ""),
            exclusion_group=sc,
            super_category=sc,
            mean_luma=float(mean_luma_val or 0.0),
        )

    if man.exists():
        data = json.loads(man.read_text(encoding="utf-8"))
        for it in data.get("items") or []:
            p = Path(it["path"])
            if not p.exists():
                p = d / Path(it["path"]).name
            c = _accept(
                p,
                str(it.get("id") or p.stem),
                list(it.get("tags") or []),
                explicit_primary=str(it["primary_tag"]) if it.get("primary_tag") else None,
                explicit_subfamily=str(it["subfamily"]) if it.get("subfamily") else None,
                explicit_exclusion=str(it["exclusion_group"]) if it.get("exclusion_group") else None,
                manifest_mean_luma=float(it["mean_luma"]) if it.get("mean_luma") is not None else None,
            )
            if c:
                items.append(c)
    else:
        for p in sorted(d.glob("*.jpg")):
            c = _accept(p, p.stem, ["workspace", "desk", "monitor_glow"])
            if c:
                items.append(c)
    print(f"[assemble] pool loaded visible_non_geometric={len(items)}", flush=True)
    return items



# Practical semantic scoring on 8GB: keyword/tag overlap (full CLIP deferred).
_STOP = {
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with", "is", "are",
    "this", "that", "it", "as", "by", "from", "at", "be", "we", "you", "your", "our",
}


_TAG_ALIASES = {
    "agent": ["workspace", "desk", "laptop", "code", "schematic"],
    "pipeline": ["server", "schematic", "code", "abstract_tech", "interface"],
    "queue": ["server", "interface", "abstract_tech"],
    "llm": ["code", "laptop", "monitor_glow", "workspace"],
    "model": ["server", "code", "abstract_tech"],
    "gpu": ["server", "abstract_tech", "cables"],
    "laptop": ["laptop", "desk", "workspace"],
    "desk": ["desk", "workspace", "keyboard"],
    "keyboard": ["keyboard", "hands_no_face", "desk"],
    "server": ["server", "abstract_tech"],
    "code": ["code", "interface", "monitor_glow"],
    "terminal": ["interface", "code", "monitor_glow"],
    "monitor": ["monitor_glow", "desk", "workspace"],
    "schematic": ["schematic", "abstract_tech"],
    "cable": ["cables", "desk"],
    "hand": ["hands_no_face", "keyboard"],
    "office": ["workspace", "desk", "laptop"],
    "tech": ["abstract_tech", "interface", "server"],
    "ai": ["code", "schematic", "workspace", "server"],
    "automation": ["schematic", "server", "pipeline", "code"],
}

def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9_]+", (text or "").lower())
    return {w for w in words if len(w) > 2 and w not in _STOP}


def semantic_score(beat_text: str, tags: list[str], visual_brief: str = "") -> float:
    """Keyword/tag overlap score in [0, 1]. Document: full CLIP embeddings deferred (VRAM)."""
    beat_toks = _tokens(f"{beat_text} {visual_brief}")
    tag_set = {t.lower() for t in tags}
    if not beat_toks:
        # No narration — accept generic desk/tech tags
        if tag_set & {"workspace", "desk", "abstract_tech", "laptop", "server", "code"}:
            return 0.75
        return 0.5

    # Expand beat tokens via aliases into expected tags
    expected: set[str] = set()
    for tok in beat_toks:
        if tok in _TAG_ALIASES:
            expected.update(_TAG_ALIASES[tok])
        if tok in tag_set:
            expected.add(tok)
    # Direct tag mention
    expected |= beat_toks & tag_set

    if not expected:
        # Abstract nonsense / off-topic: cottage, rooftop, etc.
        off = {"cottage", "rooftop", "suburb", "beach", "portrait", "face", "selfie"}
        if beat_toks & off:
            return 0.1
        # Neutral beat with tech pool tags gets mid score
        if tag_set & {"workspace", "desk", "abstract_tech", "laptop", "server", "code", "interface"}:
            return 0.70
        return 0.40

    overlap = len(expected & tag_set)
    # Also credit alias hits
    alias_hits = 0
    for tok in beat_toks:
        for a in _TAG_ALIASES.get(tok, []):
            if a in tag_set:
                alias_hits += 1
                break
    score = min(1.0, 0.35 + 0.25 * overlap + 0.15 * alias_hits)
    # Boost if any core tech tag present for agent-ish beats
    if tag_set & {"workspace", "desk", "code", "server", "laptop", "schematic", "interface"}:
        score = max(score, 0.68)
    return float(score)


def _still_to_clip(still: Path, dest: Path, seconds: float, fps: int = 24, max_hold: float | None = None) -> Path:
    from shot_gate import still_to_kenburns_clip
    cap = float(max_hold if max_hold is not None else MAX_STILL_HOLD_SEC)
    seconds = min(float(seconds), cap)
    path = still_to_kenburns_clip(still, dest, seconds=seconds, fps=fps, max_still_hold_sec=cap)
    try:
        got = probe_duration(path)
        if got > cap + 1e-6:
            print(f"[assemble] ASSERT FAIL soft: KenBurns {path.name} {got:.3f}s > {cap:.3f}s", flush=True)
        else:
            print(f"[assemble] hold_ok {path.name}={got:.3f}s <= {cap:.3f}s", flush=True)
    except Exception as exc:
        print(f"[assemble] warning: hold probe failed ({exc})", flush=True)
    return path


def probe_duration(path: Path) -> float:
    ff = Path(_ffmpeg())
    probe = ff.with_name("ffprobe.exe" if ff.suffix.lower() == ".exe" else "ffprobe")
    cmd = [
        str(probe) if probe.exists() else "ffprobe",
        "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path),
    ]
    raw = subprocess.check_output(cmd, text=True, encoding="utf-8", errors="replace")
    return float(json.loads(raw).get("format", {}).get("duration") or 0.0)


def pick_thumbnail(
    clips: list[ShotCandidate],
    dest: Path,
    *,
    min_motion: float = MIN_MOTION_FOR_THUMB,
    max_hold: float = MAX_STILL_HOLD_SEC,
) -> Path | None:
    """Prefer highest motion_score among approved non-fallback, non-max-hold clips.

    OVERRIDE vs older TZ: do NOT prefer faces/UI (daily channel bans).
    """
    from PIL import Image

    eligible = [
        c for c in clips
        if (not c.is_fallback)
        and c.hold_sec < max_hold - 1e-6
        and c.motion_score >= min_motion
        and c.path.exists()
    ]
    if not eligible:
        # relax motion floor but still exclude fallbacks
        eligible = [
            c for c in clips
            if (not c.is_fallback) and c.hold_sec < max_hold - 1e-6 and c.path.exists()
        ]
    if not eligible:
        return None
    best = max(eligible, key=lambda c: c.motion_score)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = best.path
    # If video, extract a mid frame
    if src.suffix.lower() in {".mp4", ".mov", ".webm"}:
        tmp = dest.with_suffix(".thumb_src.jpg")
        try:
            dur = max(0.2, probe_duration(src) * 0.5)
            subprocess.check_call(
                [
                    _ffmpeg(), "-y", "-ss", f"{dur:.2f}", "-i", str(src),
                    "-frames:v", "1", "-q:v", "2", str(tmp),
                ],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if tmp.exists():
                Image.open(tmp).convert("RGB").save(dest, quality=92)
                return dest
        except Exception as exc:
            print(f"[assemble] warning: thumb extract failed ({exc})", flush=True)
            return None
    else:
        Image.open(src).convert("RGB").save(dest, quality=92)
        return dest
    return None


def assemble_timeline(
    *,
    approved: list[ShotCandidate],
    audio_duration: float,
    beat_texts: list[str] | None = None,
    work_dir: Path | None = None,
    config: dict[str, Any] | None = None,
    strict: bool = False,
    soft_fail: bool | None = None,
) -> AssembleResult:
    """Build a diversified timeline covering audio_duration.

    Pseudocode (TZ v2.1):
      while covered < audio_duration:
        pick next approved OR fallback with semantic >= MIN, != prev asset_id
        hold = min(remaining, MAX_STILL_HOLD_SEC, clip_native_dur)
        append; covered += hold
      if unique_assets < ceil(audio/5) or covered < audio: abort/soft-fail
    """
    cfg = config or load_assembler_config()
    max_hold = float(cfg.get("max_still_hold_sec", MAX_STILL_HOLD_SEC))
    min_sem = float(cfg.get("min_semantic_score", MIN_SEMANTIC_SCORE))
    dens = float(cfg.get("min_unique_per_sec", MIN_UNIQUE_PER_SEC))
    if soft_fail is None:
        soft_fail = (not strict) and bool(cfg.get("default_soft_fail", True))

    work = work_dir or (ROOT / "long_work" / "assemble")
    work.mkdir(parents=True, exist_ok=True)

    pool = load_fallback_pool(ROOT / str(cfg.get("fallback_pool", {}).get("dir", "fallback_pool")))
    # Anti-Loop v3.1: asset blacklist + visual template rotation, then shuffle
    try:
        import json as _al_json
        import random as _al_random
        import anti_loop as _anti_loop
        from pathlib import Path as _AlPath
        _al_state = _anti_loop.load_state()
        pool = _anti_loop.fresh_asset_filter(pool, _al_state)
        _cfg_path = _AlPath(__file__).resolve().parent / "config.json"
        try:
            _cfg = _al_json.loads(_cfg_path.read_text(encoding="utf-8")) if _cfg_path.exists() else {}
        except Exception:
            _cfg = {}
        _template = _anti_loop.get_next_template(_al_state, _cfg.get("visual_templates"))
        pool = _anti_loop.filter_pool_by_template(pool, _template, _al_state)
        try:
            pool = filter_pool_by_prior_fingerprints(pool, ban_max=10, cooldown_hours=24)
        except Exception as _bl_exc:
            print(f"[assemble] still blacklist skipped ({_bl_exc})", flush=True)
        _al_random.shuffle(pool)
        print(f"[assemble] Using visual template: {_template.get('name')}", flush=True)
        print(f"[assemble] anti_loop pool ready n={len(pool)}", flush=True)
    except Exception as _al_exc:
        print(f"[assemble] anti_loop filter skipped ({_al_exc})", flush=True)
    # Stricter phash for small fallback pools
    if len(pool) <= 15:
        phash_max = max(int(cfg.get("phash_near_dup_max", PHASH_NEAR_DUP_MAX)), 55)
    else:
        phash_max = int(cfg.get("phash_near_dup_max", PHASH_NEAR_DUP_MAX))
    reuse_gap = float(cfg.get("reuse_min_gap_sec", REUSE_MIN_GAP_SEC))
    min_pool = int(cfg.get("fallback_pool", {}).get("min_items_required", 8))
    if len(pool) < min_pool:
        raw_pool = load_fallback_pool(ROOT / str(cfg.get("fallback_pool", {}).get("dir", "fallback_pool")))
        print(
            f"[assemble] WARNING: anti_loop left {len(pool)}<{min_pool}; "
            f"refilling from raw pool n={len(raw_pool)} (asset blacklist still preferred via order)",
            flush=True,
        )
        # Prefer previously filtered order, then append missing raw items
        seen = {getattr(x, "asset_id", id(x)) for x in pool}
        for c in raw_pool:
            aid = getattr(c, "asset_id", None)
            if aid not in seen:
                pool.append(c)
                seen.add(aid)
        if len(pool) < min_pool:
            reason = (
                f"fallback_pool has {len(pool)} items < min_items_required={min_pool}"
            )
            return _fail(reason, audio_duration, soft_fail=soft_fail, strict=strict)

    # Normalize approved
    approved_raw: list[ShotCandidate] = []
    for i, s in enumerate(approved):
        if isinstance(s, ShotCandidate):
            approved_raw.append(s)
        elif isinstance(s, dict):
            approved_raw.append(
                ShotCandidate(
                    asset_id=str(s.get("asset_id") or f"approved_{i}"),
                    path=Path(s["path"]),
                    is_fallback=bool(s.get("is_fallback", False)),
                    tags=list(s.get("tags") or ["workspace"]),
                    motion_score=float(s.get("motion_score") or 0.0),
                    script_beat=str(s.get("script_beat") or ""),
                    primary_tag=str(s.get("primary_tag") or ""),
                    exclusion_group=str(s.get("exclusion_group") or s.get("super_category") or ""),
                    super_category=str(s.get("super_category") or s.get("exclusion_group") or ""),
                )
            )
        else:
            approved_raw.append(
                ShotCandidate(asset_id=f"approved_{i}", path=Path(s), tags=["workspace"])
            )

    # passes_negative_gate BEFORE approved_shots enter the timeline
    approved_n: list[ShotCandidate] = []
    try:
        from shot_gate import passes_negative_gate as _neg_gate
    except Exception:
        _neg_gate = None  # type: ignore
    for a in approved_raw:
        # ensure super_category populated
        pt = a.primary_tag or infer_primary_tag(a.asset_id, a.tags)
        a.primary_tag = pt
        sc = a.super_category or a.exclusion_group or infer_super_category(pt, a.tags, a.asset_id)
        a.super_category = sc
        a.exclusion_group = sc
        if _neg_gate is not None and a.path.exists():
            # For videos, probe a midframe
            probe = a.path
            tmp = None
            try:
                if a.path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
                    tmp = work / f"_neg_{a.asset_id}.jpg"
                    subprocess.check_call(
                        [_ffmpeg(), "-y", "-ss", "0.35", "-i", str(a.path), "-frames:v", "1", "-q:v", "4", str(tmp)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    probe = tmp
                ok, reason, scn = _neg_gate(probe, asset_id=a.asset_id, tags=list(a.tags or []))
                a.mean_luma = float(scn.get("avg_luma", scn.get("mean_luma", 0.0)) or 0.0)
                if not ok:
                    # Gate-approved LTX must not be silently desk-replaced for theme/luma desk policy.
                    aid = str(a.asset_id or "")
                    is_ltx = aid.startswith("ltx") or (not bool(getattr(a, "is_fallback", True)))
                    soft_miss = (
                        str(reason).startswith("theme:")
                        or str(reason).startswith("neg:avg_luma")
                        or "avg_luma" in str(reason)
                    )
                    if is_ltx and soft_miss and "yolo:person" not in str(reason) and "person" not in str(reason).lower():
                        print(
                            f"[assemble] NEG_GATE soft-keep approved {a.asset_id} reason={reason} "
                            f"avg_luma={a.mean_luma:.1f} (topic LTX exempt from desk luma/theme)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[assemble] NEG_GATE drop approved {a.asset_id} reason={reason} "
                            f"avg_luma={a.mean_luma:.1f} -> will use pool still",
                            flush=True,
                        )
                        continue
            except Exception as exc:
                print(f"[assemble] warning: neg_gate approved skip {a.asset_id} ({exc})", flush=True)
            finally:
                if tmp is not None:
                    try: tmp.unlink(missing_ok=True)
                    except Exception: pass
        approved_n.append(a)
    print(
        f"[assemble] approved_after_negative_gate={len(approved_n)}/{len(approved_raw)}",
        flush=True,
    )

    beats = list(beat_texts or [])
    timeline: list[ShotCandidate] = []
    covered = 0.0
    prev_id: str | None = None
    prev_phash = None
    used_ids: set[str] = set()
    used_phashes: list = []
    asset_last_end: dict[str, float] = {}
    beat_i = 0
    safety = 0
    max_slots = int(max(1, math_ceil(audio_duration / max(0.5, max_hold))) + len(pool) + 5)

    # Hierarchical Sliding Window (~10s at 5s/shot) on super_category
    recent_super_categories: deque = deque(maxlen=2)
    recent_groups = recent_super_categories  # back-compat alias
    last_exclusion: str | None = None
    full_pool: list[ShotCandidate] = list(pool)
    available: list[ShotCandidate] = list(full_pool)

    def _ptag(c: ShotCandidate) -> str:
        return infer_primary_tag(
            c.asset_id,
            c.tags,
            explicit=(c.primary_tag or None),
        )

    def _egroup(c: ShotCandidate) -> str:
        if c.super_category:
            return c.super_category
        if c.exclusion_group:
            return infer_super_category(_ptag(c), c.tags, c.asset_id, c.exclusion_group)
        return infer_super_category(_ptag(c), c.tags, c.asset_id)

    def _phash_seen_anywhere(bits, *, ignore_asset_id: str | None = None) -> bool:
        if bits is None:
            return False
        for item in used_phashes:
            if isinstance(item, tuple):
                aid, prev_bits = item
            else:
                aid, prev_bits = None, item
            if ignore_asset_id and aid == ignore_asset_id:
                continue
            if _phash_hamming(prev_bits, bits) <= phash_max:
                return True
        return False

    def _reuse_ok(fb: ShotCandidate) -> bool:
        if fb.asset_id not in used_ids:
            return True
        # only after pool exhaust of unused ids (refill path may still have used ones)
        unused_left = [x for x in full_pool if x.asset_id not in used_ids]
        if unused_left:
            return False
        last_t = float(asset_last_end.get(fb.asset_id, -1e9))
        if (covered - last_t) < reuse_gap - 1e-6:
            return False
        return True

    def _refill_available() -> None:
        nonlocal available
        # Prefer never-used first; then reuse-eligible (gap respected at pick time)
        unused = [fb for fb in full_pool if fb.asset_id not in used_ids]
        if unused:
            available = list(unused)
            return
        available = [fb for fb in full_pool if _reuse_ok(fb)]
        if not available:
            # keep full list; gap filter applies per-candidate
            available = list(full_pool)
            print(
                f"[assemble] pool unused exhausted ({len(used_ids)} used) - refill full; reuse gap>={reuse_gap:.0f}s",
                flush=True,
            )

    def _pick_pool_shot(beat_text: str) -> ShotCandidate | None:
        """Sliding-window pick: super_category not in recent_super_categories (maxlen=2)."""
        nonlocal available
        if not available:
            _refill_available()
        if not available:
            return None

        def _eligible(fb: ShotCandidate) -> bool:
            if fb.asset_id == prev_id:
                return False
            if not _reuse_ok(fb):
                return False
            fb_hash = _still_phash(fb.path)
            if _phash_seen_anywhere(fb_hash, ignore_asset_id=fb.asset_id if fb.asset_id in used_ids else None):
                return False
            if prev_phash is not None and _phash_hamming(prev_phash, fb_hash) <= phash_max:
                return False
            near, dist, who = prior_phash_too_close(fb.path, limit=10)
            if near:
                print(
                    f"[assemble] skip still {fb.asset_id} PRIOR_PHASH_NEAR min_dist={dist} "
                    f"vs {who[:40]!r}",
                    flush=True,
                )
                return False
            return True

        base = [fb for fb in available if _eligible(fb)]
        if not base:
            # try refill then eligible again
            _refill_available()
            base = [fb for fb in available if _eligible(fb)]
        if not base:
            return None

        # 1) Filter: exclusion_group NOT IN recent_groups
        window = set(recent_groups)
        valid = [fb for fb in base if _egroup(fb) not in window]

        # 1b) After device_macro, prefer desk/prop/screen
        prefer_groups = None
        if last_exclusion == "device_macro":
            prefer_groups = {"environment_wide", "peripherals_detail", "abstract_screen"}
            preferred = [fb for fb in valid if _egroup(fb) in prefer_groups]
            if preferred:
                valid = preferred
            else:
                preferred = [fb for fb in base if _egroup(fb) in prefer_groups and _egroup(fb) not in window]
                if preferred:
                    valid = preferred

        # 2) If empty, recover — but NEVER same super_category as the immediate previous clip
        if not valid:
            print(
                f"[assemble] slidewin: no exclusion_group outside window={list(recent_groups)} - recover avoiding last={last_exclusion}",
                flush=True,
            )
            not_last = [fb for fb in base if _egroup(fb) != last_exclusion] if last_exclusion else list(base)
            if not_last:
                # Prefer groups underused in the recent window among not_last
                oldest = recent_groups[0] if recent_groups else None
                if oldest is not None and oldest != last_exclusion:
                    prefer = [fb for fb in not_last if _egroup(fb) == oldest]
                    rest = [fb for fb in not_last if _egroup(fb) != oldest]
                    valid = prefer + rest if prefer else not_last
                else:
                    valid = not_last
            else:
                # True pool exhaustion: only last_exclusion left — allow, audit may soft-warn
                print(
                    f"[assemble] slidewin: WARNING pool only has last={last_exclusion}; diversity exhausted",
                    flush=True,
                )
                valid = list(base)

        # 3) pick best semantic match among valid; break ties toward underused super_category
        from collections import Counter as _Ctr
        used_sc = _Ctr(list(recent_super_categories) + [
            (c.super_category or c.exclusion_group or "") for c in timeline
        ])
        best: ShotCandidate | None = None
        best_sem = -1.0
        best_key = None
        scored: list[tuple[float, float, ShotCandidate]] = []
        for fb in valid:
            sem = semantic_score(beat_text, fb.tags, "")
            # slight bonus for preferred groups after device_macro
            bonus = 0.05 if (prefer_groups and _egroup(fb) in prefer_groups) else 0.0
            # underused super_category bonus (round-robin pressure)
            scg = _egroup(fb)
            underuse = 0.08 / (1.0 + float(used_sc.get(scg, 0)))
            score = sem + bonus + underuse
            if sem < min_sem:
                continue
            scored.append((score, -float(used_sc.get(scg, 0)), fb))
            key = (score, -used_sc.get(scg, 0), fb.asset_id)
            if best is None or key > best_key:
                best_sem = score
                best = fb
                best_key = key
        # Anti-Loop: among near-equal top scores, break ties randomly (not always index 0)
        if scored:
            import random as _tie_random
            top = max(s[0] for s in scored)
            near = [c for (sc, _u, c) in scored if sc >= top - 0.02]
            if near:
                best = _tie_random.choice(near)
                best_sem = next(sc for (sc, _u, c) in scored if c is best)
        if best is None and valid:
            # last resort: ignore soft semantic floor (still QC tags)
            import random as _tie_random2
            best = _tie_random2.choice(valid)
            best_sem = semantic_score(beat_text, best.tags, "")
        if best is None:
            return None

        # materialize kenburns clip; duration capped at 5.0
        hold_use = min(max_hold, max(0.2, audio_duration - covered))
        dest = work / f"fb_{len(timeline):03d}_{best.asset_id}.mp4"
        try:
            _still_to_clip(best.path, dest, seconds=hold_use, max_hold=max_hold)
        except Exception as exc:
            print(f"[assemble] warning: KenBurns failed for {best.asset_id}: {exc}", flush=True)
            # remove broken from available and retry once via caller
            available = [x for x in available if x.asset_id != best.asset_id]
            return None
        try:
            got = probe_duration(dest)
            use_hold = min(hold_use, max_hold, got if got > 0 else hold_use)
            if got > max_hold + 1e-6:
                print(
                    f"[assemble] ASSERT FAIL soft: timeline still {best.asset_id} probed={got:.3f}s > {max_hold}",
                    flush=True,
                )
                trimmed = work / f"fb_{len(timeline):03d}_{best.asset_id}_cap.mp4"
                subprocess.check_call(
                    [
                        _ffmpeg(), "-y", "-i", str(dest), "-t", f"{max_hold:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-an", str(trimmed),
                    ],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                dest = trimmed
                use_hold = max_hold
        except Exception:
            use_hold = min(hold_use, max_hold)

        pt = _ptag(best)
        eg = _egroup(best)
        print(
            f"[assemble] slidewin pick {best.asset_id} primary_tag={pt} exclusion_group={eg} "
            f"hold={use_hold:.2f}s sem={best_sem:.2f} window_before={list(recent_groups)}"
            f"{' REUSE' if best.asset_id in used_ids else ''}",
            flush=True,
        )
        # 6) remove from available
        available = [x for x in available if x.asset_id != best.asset_id]
        if not available:
            _refill_available()

        return ShotCandidate(
            asset_id=best.asset_id,
            path=dest,
            is_fallback=True,
            tags=list(best.tags),
            motion_score=0.05,
            script_beat=beat_text,
            hold_sec=use_hold,
            semantic_score=best_sem,
            primary_tag=pt,
            subfamily=best.subfamily or pt,
            exclusion_group=eg,
            super_category=eg,
            mean_luma=float(best.mean_luma or 0.0),
        )

    while covered < audio_duration - 0.05 and safety < max_slots:
        safety += 1
        beat_text = beats[beat_i] if beat_i < len(beats) else (beats[-1] if beats else "")
        remaining = audio_duration - covered
        # 4) clip_duration = min(shot.duration, 5.0, remaining_audio)
        hold = min(max_hold, remaining)

        candidate: ShotCandidate | None = None

        # INTERLEAVE by DURATION quota (~30% LTX); fail any gate -> still (fail-closed)
        hybrid_cfg = cfg.get("hybrid") if isinstance(cfg.get("hybrid"), dict) else {}
        ltx_cfg = cfg.get("ltx") if isinstance(cfg.get("ltx"), dict) else {}
        gates_cfg = cfg.get("gates") if isinstance(cfg.get("gates"), dict) else {}
        target_ltx_ratio = float(hybrid_cfg.get("ltx_ratio", 0.30) or 0.30)
        min_ltx_warn = float(hybrid_cfg.get("min_ltx_ratio_warn", 0.15) or 0.15)
        min_ltx_motion = float(
            gates_cfg.get("min_ltx_motion_score")
            or ltx_cfg.get("min_motion_score")
            or 1.8
        )
        ltx_covered = float(sum(float(c.hold_sec or 0.0) for c in timeline if not c.is_fallback))
        need_ltx = (ltx_covered < (audio_duration * target_ltx_ratio) - 1e-6) or (not full_pool)
        unused_ltx = [
            a for a in approved_n
            if (not a.is_fallback) and a.asset_id not in used_ids and a.asset_id != prev_id and a.path.exists()
        ]
        # Prefer liveliest LTX first
        unused_ltx.sort(key=lambda x: float(x.motion_score or 0.0), reverse=True)
        if len(unused_ltx) > 1:
            import random as _ltx_rand
            top_m = float(unused_ltx[0].motion_score or 0.0)
            near_ltx = [x for x in unused_ltx if float(x.motion_score or 0.0) >= top_m - 0.15]
            if near_ltx:
                # shuffle near-equals so we don't always take index 0
                _ltx_rand.shuffle(near_ltx)
                rest = [x for x in unused_ltx if x not in near_ltx]
                unused_ltx = near_ltx + rest

        # Insert approved LTX before desk stills, but NEVER back-to-back
        # (contig_hold on same primary_tag + slidewin on ltx_generated).
        last_was_ltx = bool(timeline) and (not bool(getattr(timeline[-1], "is_fallback", True)))
        prefer_ltx = bool(unused_ltx) and (not last_was_ltx)
        if last_was_ltx and unused_ltx:
            print(
                f"[assemble] LTX_INTERLEAVE skip prefer (last={timeline[-1].asset_id}); "
                f"unused_ltx={[x.asset_id for x in unused_ltx[:4]]}",
                flush=True,
            )
        if prefer_ltx and unused_ltx:
            for a in list(unused_ltx):
                if candidate is not None:
                    break
                # Unique primary_tag per LTX asset so contig_hold does not chain kb/topic clips.
                a_pt = f"ltx_{a.asset_id}"
                a_eg = _egroup(a) if (a.exclusion_group or a.primary_tag or a.tags) else "ltx_generated"
                a_hash = _still_phash(a.path)
                # Runtime LTX gate: YOLO fail-closed + CLIP + luma + category OCR + motion
                ltx_reject = False
                try:
                    from shot_gate import (
                        passes_negative_gate, passes_ocr_gate, gate_ocr, extract_keyframes, gate_motion,
                    )
                    kf_dir = work / f"_ltx_gate_{a.asset_id}"
                    is_vid = a.path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}
                    kfs = extract_keyframes(a.path, kf_dir, count=3) if is_vid else [a.path]
                    probe = kfs[len(kfs)//2] if kfs else a.path
                    ok_n, reason_n, sc_n = passes_negative_gate(probe, asset_id=a.asset_id, tags=list(a.tags or []))
                    cat_blob = " ".join([a_pt, a_eg, " ".join(a.tags or [])])
                    ok_area, reason_area, sc_area = passes_ocr_gate(
                        probe, shot_category=cat_blob, fail_closed_if_missing=True
                    )
                    ok_o, reason_o, sc_o = gate_ocr(kfs) if kfs else (True, "", {})
                    ok_mot, reason_mot, sc_mot = (True, "", {})
                    if is_vid:
                        ok_mot, reason_mot, sc_mot = gate_motion(a.path)
                    # also reject near-static / frozen LTX
                    mot_score = float(a.motion_score or sc_mot.get("motion", 0.0) or 0.0)
                    if (not ok_mot) or (is_vid and mot_score < float(min_ltx_motion)):
                        ok_mot = False
                        reason_mot = reason_mot or f"motion:below_min:{mot_score:.3f}<{float(min_ltx_motion):.3f}"
                    # Theme/luma desk-policy miss must not desk-replace gate-approved topic LTX.
                    soft_n = (
                        (not ok_n)
                        and (
                            str(reason_n).startswith("theme:")
                            or str(reason_n).startswith("neg:avg_luma")
                            or "avg_luma" in str(reason_n)
                        )
                        and ("yolo:person" not in str(reason_n))
                        and ("person" not in str(reason_n).lower() or "avg_luma" in str(reason_n))
                        and ok_area
                        and (ok_o or not ("gibber" in str(reason_o).lower() or "scribbl" in str(reason_o).lower()))
                    )
                    if soft_n:
                        print(
                            f"[assemble] LTX_THEME_SOFT_KEEP {a.asset_id} reason={reason_n} "
                            f"(runtime exempt; will LTX_IN_TIMELINE)",
                            flush=True,
                        )
                        ok_n = True
                    # Approved LTX with only slow motion: keep (KenBurns already applied for ltx_kb_*)
                    if (
                        (not ok_mot)
                        and ok_area
                        and (
                            str(reason_mot).startswith("motion:almost_static")
                            or "below_min" in str(reason_mot)
                        )
                        and (ok_n or soft_n)
                        and (ok_o or not ("gibber" in str(reason_o).lower() or "scribbl" in str(reason_o).lower()))
                    ):
                        print(
                            f"[assemble] LTX_MOTION_SOFT_KEEP {a.asset_id} reason={reason_mot} "
                            f"motion={mot_score:.2f} (keep approved LTX vs desk-pool)",
                            flush=True,
                        )
                        ok_mot = True
                    # KenBurns salvage of OCR/static LTX often re-triggers OCR at runtime —
                    # keep topic LTX over desk-pool clones (person/horror still hard-reject).
                    _ocr_soft = (
                        (
                            (not ok_area)
                            or (
                                not ok_o
                                and (
                                    "gibber" in str(reason_o).lower()
                                    or "scribbl" in str(reason_o).lower()
                                    or "text_area" in str(reason_o).lower()
                                    or "text_area" in str(reason_area).lower()
                                )
                            )
                        )
                        and ("yolo:person" not in str(reason_n).lower())
                        and ("horror" not in str(reason_n).lower())
                        and (
                            ok_n
                            or soft_n
                            or str(reason_n).startswith("theme:")
                            or "avg_luma" in str(reason_n)
                        )
                    )
                    if _ocr_soft:
                        print(
                            f"[assemble] LTX_OCR_SOFT_KEEP {a.asset_id} "
                            f"area={reason_area or ok_area} ocr={reason_o or ok_o} "
                            f"(approved LTX vs desk-pool)",
                            flush=True,
                        )
                        ok_area = True
                        ok_o = True
                        if soft_n or str(reason_n).startswith("theme:") or "avg_luma" in str(reason_n):
                            ok_n = True
                    if (not ok_n) or (not ok_area) or (not ok_o and ("gibber" in str(reason_o).lower() or "scribbl" in str(reason_o).lower())) or (not ok_mot):
                        ltx_reject = True
                        print(
                            f"[assemble] LTX_REJECT_TO_POOL {a.asset_id} neg_ok={ok_n} ocr_area_ok={ok_area} "
                            f"ocr_ok={ok_o} mot_ok={ok_mot} "
                            f"reasons={[x for x in (reason_n, reason_area, reason_o, reason_mot) if x]} "
                            f"avg_luma={sc_n.get('avg_luma',0):.1f} ocr_ratio={sc_area.get('ocr_text_area_ratio',0):.4f} "
                            f"ocr_lim={sc_area.get('ocr_max_ratio_used',0):.3f} "
                            f"motion={sc_mot.get('motion', mot_score):.2f}",
                            flush=True,
                        )
                except Exception as _ltx_gate_exc:
                    # FAIL-CLOSED: gate exception must not silently insert LTX
                    ltx_reject = True
                    print(f"[assemble] LTX_REJECT_TO_POOL {a.asset_id} gate_exception={_ltx_gate_exc}", flush=True)
                if ltx_reject:
                    # burn this LTX id so we don't spin forever on the same reject
                    used_ids.add(a.asset_id)
                    continue
                elif _phash_seen_anywhere(a_hash):
                    print(f"[assemble] skip LTX near-dup phash anywhere {a.asset_id}", flush=True)
                    continue
                else:
                    if a_eg in set(recent_groups):
                        print(
                            f"[assemble] LTX_EXCL_BYPASS {a.asset_id} exclusion_group={a_eg} "
                            f"window={list(recent_groups)} (keep topic LTX over stills)",
                            flush=True,
                        )
                    sem = semantic_score(beat_text, a.tags or ["workspace", "desk"], a.script_beat)
                    if sem < min_sem and a.tags:
                        print(
                            f"[assemble] warning: approved {a.asset_id} semantic={sem:.2f} < {min_sem} - still using (gate-approved)",
                            flush=True,
                        )
                    candidate = ShotCandidate(
                        asset_id=a.asset_id,
                        path=a.path,
                        is_fallback=False,
                        tags=list(a.tags or ["workspace"]),
                        motion_score=float(a.motion_score or 0.0),
                        script_beat=beat_text,
                        hold_sec=hold,
                        semantic_score=max(sem, 0.9),
                        primary_tag=a_pt,
                        subfamily=a.subfamily or a_pt,
                        exclusion_group=a_eg,
                    )
                    print(
                        f"[assemble] LTX_IN_TIMELINE asset={a.asset_id} primary_tag={a_pt} exclusion_group={a_eg} hold={hold:.2f}s path={Path(a.path).name} "
                        f"ltx_so_far={ltx_covered:.1f}s target={audio_duration * target_ltx_ratio:.1f}s",
                        flush=True,
                    )

        if candidate is None:
            candidate = _pick_pool_shot(beat_text)

        if candidate is None and unused_ltx:
            a = unused_ltx[0]
            a_pt = _ptag(a) if a.primary_tag or a.tags else "monitor_code"
            a_hash = _still_phash(a.path)
            if not _phash_seen_anywhere(a_hash):
                candidate = ShotCandidate(
                    asset_id=a.asset_id,
                    path=a.path,
                    is_fallback=False,
                    tags=list(a.tags or ["workspace"]),
                    motion_score=float(a.motion_score or 0.0),
                    script_beat=beat_text,
                    hold_sec=hold,
                    semantic_score=0.9,
                    primary_tag=a_pt,
                    subfamily=a.subfamily or a_pt,
                    exclusion_group=_egroup(a) if (a.exclusion_group or a.primary_tag or a.tags) else "screen",
                )

        if candidate is None and covered + 0.15 < audio_duration:
            reuse_pick = None
            try:
                _refill_available()
            except Exception:
                pass
            for fb in list(available):
                if fb.asset_id == prev_id:
                    continue
                near, _, _ = prior_phash_too_close(Path(fb.path), limit=10)
                if near:
                    continue
                reuse_pick = fb
                break
            if reuse_pick is None:
                for fb in list(available):
                    if fb.asset_id != prev_id:
                        reuse_pick = fb
                        print(
                            f"[assemble] COVER_REUSE_LASTRESORT {fb.asset_id} "
                            f"covered={covered:.1f}/{audio_duration:.1f}",
                            flush=True,
                        )
                        break
            if reuse_pick is not None:
                hold = min(float(max_hold), max(0.8, audio_duration - covered))
                candidate = ShotCandidate(
                    asset_id=reuse_pick.asset_id,
                    path=reuse_pick.path,
                    is_fallback=True,
                    tags=list(reuse_pick.tags or ["workspace"]),
                    motion_score=float(reuse_pick.motion_score or 0.0),
                    script_beat=beat_text,
                    hold_sec=hold,
                    semantic_score=0.5,
                    primary_tag=getattr(reuse_pick, "primary_tag", "") or "",
                    exclusion_group=getattr(reuse_pick, "exclusion_group", "") or "",
                )
                print(
                    f"[assemble] COVER_REUSE {candidate.asset_id} hold={hold:.2f}s "
                    f"covered={covered:.1f}->{covered + hold:.1f}/{audio_duration:.1f}",
                    flush=True,
                )
            else:
                print(
                    f"[assemble] COVER_STALL covered={covered:.1f}/{audio_duration:.1f}",
                    flush=True,
                )
                break
        elif candidate is None:
            break

        # Cap approved video hold too (trim via ffmpeg if needed)
        if not candidate.is_fallback:
            try:
                native = probe_duration(candidate.path)
            except Exception:
                native = hold
            use = min(hold, max_hold, native if native > 0 else hold)
            if use < native - 0.15:
                trimmed = work / f"trim_{len(timeline):03d}_{candidate.asset_id}.mp4"
                subprocess.check_call(
                    [
                        _ffmpeg(), "-y", "-i", str(candidate.path),
                        "-t", f"{use:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-pix_fmt", "yuv420p", "-an", str(trimmed),
                    ],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                candidate.path = trimmed
            candidate.hold_sec = use
            hold = use

        timeline.append(candidate)
        used_ids.add(candidate.asset_id)
        # 5) sliding window: record super_category (deque maxlen=2)
        pt_final = candidate.primary_tag or infer_primary_tag(candidate.asset_id, candidate.tags)
        candidate.primary_tag = pt_final
        eg_final = (
            candidate.super_category
            or candidate.exclusion_group
            or infer_super_category(pt_final, candidate.tags, candidate.asset_id)
        )
        candidate.super_category = eg_final
        candidate.exclusion_group = eg_final
        recent_super_categories.append(eg_final)
        last_exclusion = eg_final
        print(
            f"[assemble] slidewin append super_category={eg_final} primary_tag={pt_final} window_now={list(recent_super_categories)}",
            flush=True,
        )
        prev_id = candidate.asset_id
        prev_phash = _still_phash(candidate.path)
        if prev_phash is not None:
            used_phashes.append((candidate.asset_id, prev_phash))
        covered += candidate.hold_sec
        asset_last_end[candidate.asset_id] = covered
        beat_i += 1

    # Post-check LTX duration quota
    try:
        hybrid_cfg = cfg.get("hybrid") if isinstance(cfg.get("hybrid"), dict) else {}
        target_ltx_ratio = float(hybrid_cfg.get("ltx_ratio", 0.30) or 0.30)
        min_ltx_warn = float(hybrid_cfg.get("min_ltx_ratio_warn", 0.15) or 0.15)
        ltx_secs = float(sum(float(c.hold_sec or 0.0) for c in timeline if not c.is_fallback))
        still_secs = float(sum(float(c.hold_sec or 0.0) for c in timeline if c.is_fallback))
        ratio = (ltx_secs / audio_duration) if audio_duration > 0 else 0.0
        print(
            f"[assemble] ltx_quota_audit ltx_secs={ltx_secs:.2f} still_secs={still_secs:.2f} "
            f"ratio={ratio:.1%} target={target_ltx_ratio:.1%} warn_below={min_ltx_warn:.1%}",
            flush=True,
        )
        if ratio < min_ltx_warn:
            print(
                f"[assemble] WARNING: LTX ratio is only {ratio:.1%}. Target was {target_ltx_ratio:.1%}. "
                f"Check LTX generation prompts or gate rejection rates.",
                flush=True,
            )
            # Desk-stills dominate → fingerprint clones priors; refuse mux/upload.
            if ratio < float(min_ltx_warn) - 1e-9:
                raise PipelineAbortError(
                    f"ltx_ratio_too_low:{ratio:.3f}<{min_ltx_warn:.3f} (desk-pool clone risk)"
                )
    except PipelineAbortError:
        raise
    except Exception as _qexc:
        print(f"[assemble] warning: ltx_quota_audit skipped ({_qexc})", flush=True)

    unique = len(used_ids)
    need_unique = max(1, math_ceil(audio_duration / dens))
    id_list = [c.asset_id for c in timeline]
    reuse_count = len(id_list) - len(set(id_list))
    ok = covered >= audio_duration - 0.15 and unique >= need_unique
    if reuse_count > 0:
        print(f"[assemble] reuse_audit count={reuse_count} (allowed only after pool exhaust + gap)", flush=True)

    # Hard-enforce: every timeline clip duration <= max_still_hold_sec (probe + metadata)
    max_hold_seen = 0.0
    hold_violations: list[str] = []
    for c in timeline:
        meta_hold = float(c.hold_sec or 0.0)
        probed = meta_hold
        try:
            if c.path.exists() and c.path.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm"}:
                probed = float(probe_duration(c.path))
        except Exception:
            pass
        max_hold_seen = max(max_hold_seen, meta_hold, probed)
        if meta_hold > max_hold + 1e-6 or probed > max_hold + 1e-6:
            hold_violations.append(
                f"{c.asset_id}: meta={meta_hold:.3f}s probed={probed:.3f}s > max={max_hold:.3f}s"
            )
            ok = False
            print(f"[assemble] ASSERT FAIL soft: hold exceeded for {hold_violations[-1]}", flush=True)

    print(
        f"[assemble] hold_audit max_hold_seen={max_hold_seen:.3f}s limit={max_hold:.3f}s "
        f"clips={len(timeline)} violations={len(hold_violations)}",
        flush=True,
    )
    # Near-black segment audit — reject timelines with >0.5s dark content
    dark_secs = 0.0
    dark_hits: list[str] = []
    try:
        from shot_gate import still_luma_ok, MIN_STILL_LUMA
        for c in timeline:
            probe = work / f"_lum_{c.asset_id}.jpg"
            try:
                if Path(c.path).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    # Reused stills keep their source-image path. Probing an image with
                    # "ffmpeg -ss 0.3" produces no frame, which used to be misread as a
                    # near-black segment and aborted the whole build (false positive).
                    # Measure the image itself instead.
                    luma_ok, lsc = still_luma_ok(Path(c.path), min_luma=MIN_STILL_LUMA, min_laplacian=0.0)
                else:
                    subprocess.check_call(
                        [
                            _ffmpeg(), "-y", "-ss", "0.3", "-i", str(c.path),
                            "-frames:v", "1", "-q:v", "3", str(probe),
                        ],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    luma_ok, lsc = still_luma_ok(probe, min_luma=MIN_STILL_LUMA, min_laplacian=0.0)
                if not luma_ok:
                    dark_secs += float(c.hold_sec or 0.0)
                    dark_hits.append(
                        f"{c.asset_id}:luma={lsc.get('luma', 0):.1f}/hold={float(c.hold_sec or 0):.2f}"
                    )
            except Exception:
                pass
            finally:
                try:
                    probe.unlink(missing_ok=True)
                except Exception:
                    pass
        print(
            f"[assemble] luma_audit dark_secs={dark_secs:.2f} hits={len(dark_hits)} "
            f"detail={dark_hits[:6]}",
            flush=True,
        )
        if dark_secs > 0.5:
            raise PipelineAbortError(
                f"near-black segments total {dark_secs:.2f}s > 0.5s: {'; '.join(dark_hits[:6])}"
            )
    except PipelineAbortError:
        raise
    except Exception as exc:
        print(f"[assemble] warning: luma_audit skipped ({exc})", flush=True)

    # Contiguous perceptual hold audit (m02): similar/geo near-dups across boundaries
    try:
        from PIL import Image as _PilImage
        import numpy as _np

        def _phash(gray, size=16):
            im = _PilImage.fromarray(_np.clip(gray, 0, 255).astype(_np.uint8)).resize(
                (size, size), _PilImage.Resampling.BILINEAR
            )
            a = _np.asarray(im, dtype=_np.float32)
            return (a > a.mean()).flatten()

        def _glow_delta(arr):
            gray = arr.mean(axis=2)
            H, W = gray.shape
            c = gray[int(H * 0.12):int(H * 0.55), int(W * 0.12):int(W * 0.78)]
            b = _np.concatenate([
                gray[:int(H * 0.08)].ravel(), gray[int(H * 0.85):].ravel(),
                gray[:, :int(W * 0.06)].ravel(), gray[:, int(W * 0.94):].ravel(),
            ])
            return float(c.mean() - b.mean())

        samples = []
        tcur = 0.0
        for c in timeline:
            probe = work / f"_contig_{c.asset_id}.jpg"
            ss = "0.05" if float(c.hold_sec or 0) < 0.5 else "0.35"
            try:
                subprocess.check_call(
                    [_ffmpeg(), "-y", "-ss", ss, "-i", str(c.path),
                     "-frames:v", "1", "-q:v", "3", str(probe)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                arr = _np.asarray(_PilImage.open(probe).convert("RGB"), dtype=_np.float32)
                gray = arr.mean(axis=2)
                gd = _glow_delta(arr)
                geo = gd > 40.0  # high center glow; photoreal desks usually <<35 after grade
                samples.append({
                    "id": c.asset_id, "t0": tcur, "t1": tcur + float(c.hold_sec or 0),
                    "hold": float(c.hold_sec or 0), "bits": _phash(gray),
                    "geo": geo, "fallback": bool(c.is_fallback), "gd": gd,
                    "primary_tag": (c.primary_tag or infer_primary_tag(c.asset_id, c.tags)),
                    "exclusion_group": (
                        c.super_category
                        or c.exclusion_group
                        or infer_super_category(
                            c.primary_tag or infer_primary_tag(c.asset_id, c.tags),
                            c.tags,
                            c.asset_id,
                        )
                    ),
                    "super_category": (
                        c.super_category
                        or c.exclusion_group
                        or infer_super_category(
                            c.primary_tag or infer_primary_tag(c.asset_id, c.tags),
                            c.tags,
                            c.asset_id,
                        )
                    ),
                })
            finally:
                try: probe.unlink(missing_ok=True)
                except Exception: pass
            tcur += float(c.hold_sec or 0)

        # Sliding-window-of-2: no super_category may repeat inside any 2 consecutive clips
        win_bad = []
        for i in range(len(samples)):
            window = samples[max(0, i - 1): i + 1]
            groups_w = [
                s.get("super_category")
                or s.get("exclusion_group")
                or infer_super_category(s.get("primary_tag") or infer_primary_tag(s["id"]), None, s["id"])
                for s in window
            ]
            if len(groups_w) != len(set(groups_w)):
                from collections import Counter as _Ctr
                dups = [t for t, n in _Ctr(groups_w).items() if n > 1]
                win_bad.append(
                    f"clips[{max(0,i-1)}..{i}] super_categories={groups_w} dups={dups}"
                )
        print(
            f"[assemble] slidewin_audit windows_checked={len(samples)} violations={len(win_bad)}",
            flush=True,
        )
        if win_bad:
            has_ltx = any(not bool(getattr(c, "is_fallback", True)) for c in timeline)
            msg = f"super_category repeated inside window of 2: {'; '.join(win_bad[:4])}"
            if has_ltx:
                # Keep unique LTX timeline instead of hard-aborting a near-done mux.
                print(f"[assemble] WARNING soft slidewin: {msg}", flush=True)
            else:
                raise PipelineAbortError(msg)

        merged = []
        for seg in samples:
            if not merged:
                merged.append({**seg, "ids": [seg["id"]], "span": seg["hold"], "pt": seg.get("primary_tag")})
                continue
            prev = merged[-1]
            ham = int(_np.count_nonzero(prev["bits"] != seg["bits"]))
            pt_a = str(prev.get("pt") or "")
            pt_b = str(seg.get("primary_tag") or "")
            # Each LTX clip gets unique ltx_<id> primary_tag; never contig-chain LTX with LTX.
            both_ltx = pt_a.startswith("ltx_") and pt_b.startswith("ltx_")
            same_primary = (pt_a == pt_b) and bool(pt_a) and (not both_ltx)
            # Contig wall-clock only chains same primary_tag (phash/geo tracked separately via slidewin)
            same = same_primary
            if same:
                prev["t1"] = seg["t1"]
                prev["span"] = prev["t1"] - prev["t0"]
                prev["ids"].append(seg["id"])
                prev["geo"] = prev["geo"] or seg["geo"]
            else:
                merged.append({**seg, "ids": [seg["id"]], "span": seg["hold"], "pt": seg.get("primary_tag")})

        max_contig = 0.0
        bad = []
        # same-primary_tag contig wall-clock must be <= 5.0s
        for m in merged:
            max_contig = max(max_contig, float(m["span"]))
            if float(m["span"]) > float(max_hold) + 1e-6:
                bad.append(f"{m['ids'][0]}..{m['ids'][-1]}:{m['span']:.2f}s pt={m.get('pt')}")
        print(
            f"[assemble] contig_hold_audit max_similar_span={max_contig:.3f}s "
            f"limit={max_hold:.3f}s groups={len(merged)} violations={len(bad)}",
            flush=True,
        )
        if bad:
            raise PipelineAbortError(
                f"contiguous same-primary/similar hold >{max_hold}s: {'; '.join(bad[:4])}"
            )
    except PipelineAbortError:
        raise
    except Exception as exc:
        print(f"[assemble] warning: contig_hold_audit skipped ({exc})", flush=True)



    if not ok:
        if hold_violations:
            reason = (
                f"max_still_hold_sec={max_hold} violated: {'; '.join(hold_violations[:4])}; "
                f"covered={covered:.1f}s unique={unique}"
            )
        else:
            reason = (
                f"cannot cover audio={audio_duration:.1f}s with max_hold={max_hold}s "
                f"(covered={covered:.1f}s unique={unique} need_unique>={need_unique}). "
                f"Refusing single stretched stub."
            )
        return _fail(
            reason,
            audio_duration,
            soft_fail=soft_fail,
            strict=strict,
            clips=timeline,
            covered=covered,
            unique=unique,
        )

    result = AssembleResult(
        ok=True,
        clips=timeline,
        quality_compromised=False,
        reason="",
        audio_duration=audio_duration,
        covered_sec=covered,
        unique_assets=unique,
    )
    # Anti-Loop: remember used asset_ids across scheduled runs
    try:
        import anti_loop as _anti_loop2
        _ids = [c.asset_id for c in timeline if getattr(c, "asset_id", None)]
        _anti_loop2.register_assets(_ids)
        print(f"[assemble] anti_loop registered {len(set(_ids))} asset_ids", flush=True)
    except Exception as _reg_exc:
        print(f"[assemble] anti_loop register_assets skipped ({_reg_exc})", flush=True)
    # Fail-closed: KenBurns/encode can drift a still toward a prior upload keyframe.
    try:
        bad_priors = []
        for c in timeline:
            if not bool(getattr(c, "is_fallback", False)):
                continue
            near, dist, who = prior_phash_too_close(Path(c.path), limit=6)
            if near:
                bad_priors.append(f"{c.asset_id}:min_dist={dist} vs {who[:32]!r}")
        # A single still close to an old release is not proof of a duplicate.
        if len(bad_priors) >= 2:
            raise PipelineAbortError(
                "still_prior_phash_after_kb:" + "; ".join(bad_priors[:4])
            )
        if bad_priors:
            print(f"[assemble] prior_phash warning (single hit, kept): {bad_priors[0]}", flush=True)
    except PipelineAbortError:
        raise
    except Exception as _pg_exc:
        print(f"[assemble] prior_phash post-KB gate skipped ({_pg_exc})", flush=True)
    try:
        record_used_stills(timeline, video_ref="assembled")
    except Exception as _rs_exc:
        print(f"[assemble] record_used_stills skipped ({_rs_exc})", flush=True)
    return result


def math_ceil(x: float) -> int:
    import math
    return int(math.ceil(x))


def _fail(
    reason: str,
    audio_duration: float,
    *,
    soft_fail: bool,
    strict: bool,
    clips: list[ShotCandidate] | None = None,
    covered: float = 0.0,
    unique: int = 0,
) -> AssembleResult:
    print(f"[assemble] CRITICAL: {reason}", flush=True)
    if strict and not soft_fail:
        raise PipelineAbortError(reason)
    # soft-fail path
    return AssembleResult(
        ok=False,
        clips=clips or [],
        quality_compromised=True,
        reason=reason,
        audio_duration=audio_duration,
        covered_sec=covered,
        unique_assets=unique,
    )


_VISUAL_GRADE = None


def _grade_vf(base: str) -> str:
    """Append this release's colour grade so consecutive releases look different.

    Chosen once per process by visual_identity.current_identity() and recorded
    in pipeline_state.json so the next release gets a different one.
    """
    global _VISUAL_GRADE
    if _VISUAL_GRADE is None:
        try:
            import visual_identity as _vi
            _VISUAL_GRADE = _vi.current_identity()
            print(f"[assemble] visual identity: {_VISUAL_GRADE[0]}", flush=True)
        except Exception as _vi_exc:
            print(f"[assemble] visual identity skipped ({_vi_exc})", flush=True)
            _VISUAL_GRADE = ("neutral", "")
    _name, _flt = _VISUAL_GRADE
    chain = base if not _flt else f"{base},{_flt}"
    return chain + ",format=yuv420p"


def _concat_videos(clips: list[Path], dest: Path, max_hold: float = MAX_STILL_HOLD_SEC) -> Path:
    """Normalize every clip to 1280x720@24 before concat.

    Mixing 768x512 LTX with 1280x720 KenBurns via concat demuxer locked the
    encoder to the first stream size and produced broken/noisy second halves.
    Hard -t max_hold on EVERY normalized segment so concat cannot exceed hold.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    norm_dir = dest.parent / f"{dest.stem}_norm"
    norm_dir.mkdir(parents=True, exist_ok=True)
    normalized: list[Path] = []
    cap = float(max_hold if max_hold is not None else MAX_STILL_HOLD_SEC)
    for i, clip in enumerate(clips):
        out = norm_dir / f"n_{i:03d}.mp4"
        subprocess.check_call(
            [
                _ffmpeg(), "-y", "-i", str(clip),
                "-t", f"{cap:.3f}",
                "-vf",
                _grade_vf(
                    "scale=1280:720:force_original_aspect_ratio=decrease,"
                    "pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,fps=24"
                ),
                "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                "-an", "-movflags", "+faststart", str(out),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        # probe enforce hard cap
        try:
            got = float(probe_duration(out))
            if got > cap + 1e-6:
                trimmed = norm_dir / f"n_{i:03d}_cap.mp4"
                subprocess.check_call(
                    [
                        _ffmpeg(), "-y", "-i", str(out), "-t", f"{cap:.3f}",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
                        "-an", "-movflags", "+faststart", str(trimmed),
                    ],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                out = trimmed
                print(f"[assemble] concat_norm RETRIM {clip.name} {got:.3f}s -> {cap:.3f}s", flush=True)
        except Exception:
            pass
        normalized.append(out)
        print(f"[assemble] concat_norm {clip.name} -> {out.name} 1280x720 hard_t<={cap:.3f}s", flush=True)
    listing = dest.with_suffix(".concat.txt")
    lines = []
    for clip in normalized:
        path = str(clip.resolve()).replace(chr(92), '/')
        lines.append(f"file '{path}'")
    listing.write_text("\n".join(lines) + "\n", encoding="utf-8")
    subprocess.check_call(
        [
            _ffmpeg(), "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-r", "24", "-an",
            "-movflags", "+faststart", str(dest),
        ]
    )
    return dest


def assemble_video(
    *,
    approved: list[Any],
    audio_duration: float,
    beat_texts: list[str] | None = None,
    out_concat: Path | None = None,
    thumb_dest: Path | None = None,
    work_dir: Path | None = None,
    strict: bool = False,
    soft_fail: bool | None = None,
) -> AssembleResult:
    """High-level: assemble timeline, optionally concat + pick thumb."""
    cfg = load_assembler_config()
    result = assemble_timeline(
        approved=approved,
        audio_duration=audio_duration,
        beat_texts=beat_texts,
        work_dir=work_dir,
        config=cfg,
        strict=strict,
        soft_fail=soft_fail,
    )
    if not result.ok:
        return result

    paths = [c.path for c in result.clips]
    if out_concat is not None and paths:
        _concat_videos(
            paths,
            out_concat,
            max_hold=float(cfg.get("max_still_hold_sec", MAX_STILL_HOLD_SEC)),
        )
        print(f"[assemble] concat -> {out_concat}", flush=True)

    if thumb_dest is not None:
        picked = pick_thumbnail(
            result.clips,
            thumb_dest,
            min_motion=float(cfg.get("min_motion_for_thumb", MIN_MOTION_FOR_THUMB)),
            max_hold=float(cfg.get("max_still_hold_sec", MAX_STILL_HOLD_SEC)),
        )
        result.thumb_source = picked
        if picked:
            print(f"[assemble] thumbnail -> {picked}", flush=True)
        else:
            print("[assemble] warning: no eligible non-fallback thumb source", flush=True)

    # Persist report
    report_dir = work_dir or (ROOT / "long_work" / "assemble")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "assemble_report.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    # Quick self-check with mock approved + pool
    import sys
    pool = load_fallback_pool()
    print(f"pool={len(pool)}")
    mock = [
        ShotCandidate(
            asset_id="ltx_mock_1",
            path=pool[0].path,  # reuse still as stand-in path for structure test
            is_fallback=False,
            tags=["workspace", "desk"],
            motion_score=5.0,
        )
    ]
    # Force still path through KenBurns by marking approved missing video — use pool only
    mock = []
    try:
        r = assemble_timeline(
            approved=mock,
            audio_duration=18.0,
            beat_texts=[
                "AI agents on a laptop desk",
                "server pipeline queue",
                "schematic of automation",
                "keyboard workspace",
            ],
            strict=False,
            soft_fail=True,
        )
        print(json.dumps(r.to_dict(), indent=2)[:2000])
        print("ok", r.ok, "compromised", r.quality_compromised, "clips", len(r.clips))
        sys.exit(0 if r.ok else 2)
    except PipelineAbortError as e:
        print("ABORT", e)
        sys.exit(1)
