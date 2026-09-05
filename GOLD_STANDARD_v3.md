# GOLD STANDARD v3.0 — Winning Configuration

**Purpose / Назначение:** зафиксировать проверенный PASS-конфиг для ежедневной фабрики LTX hybrid (11:00 / 18:00) и handoff. Не трогать Autopilot.

**PASS proof:** https://youtu.be/58K_BqdM-_E  
**Density note:** ~28s dense beats > 44s padded junk. Держать visual density: `min_unique_per_sec: 5.0`, hold ≤ 5.0s.

**Loaded by code:** `config/video_assembler.yaml` via `video_assembler.load_assembler_config()`  
**Canonical twin:** `config/gold_standard_v3.yaml` (same keys)  
**Gate constants also in:** `shot_gate.py` (YOLO/CLIP/OCR/luma), `run.py` (hybrid ~30% LTX + SAFETY_NEGATIVE + cinematic pan bias)

---

## Gates table / Таблица гейтов

| Gate | Threshold | Notes |
|------|-----------|--------|
| **YOLO** person/hand/arm/face | conf ≥ **0.15** → reject | other classes **0.25** if applicable |
| **CLIP allowlist** | max sim ≥ **0.22** | desk, technology, computer monitor, keyboard, workspace, dark office |
| **OCR** (RapidOCR) | text area ratio ≤ **0.05** | brand logos on hardware OK if small |
| **Dark** | avg_luma ≤ **100** | Laplacian insurance `min_laplacian_var: 100` if present |
| **Rhythm slidewin** | deque **maxlen=2** on `super_category` | no repeat inside window of 2 |
| **Hold** | ≤ **5.0s** | hard cap on stills / KenBurns / concat |
| **phash** | near-dup ban anywhere | `phash_near_dup_max: 55` |
| **Reuse** | global `used_ids`; gap ≥ **30s** after pool exhaust | never reuse while unused remain |

---

## Hybrid rules / Гибрид

- **~70% stills / ~30% LTX** (slot pattern: 1 LTX per ~3–4 stills).
- LTX проходит **те же гейты** (YOLO + CLIP + luma + OCR + motion).
- LTX fail → **still той же / совместимой `super_category`** (pool pick + slidewin).
- LTX negatives: people, hands, faces, text, ui, bright, white, daylight.
- Prompt bias: **slow cinematic pan** / dark tech desk (parallax, moody).
- Brand logos on hardware — OK (малый текст / OCR area).

---

## Daily checklist / Ежедневный чеклист (11:00 & 18:00)

1. ComfyUI + Ollama up (`run_scheduled.ps1` стартует при необходимости).
2. Pool: `fallback_pool/` ≥ `min_items_required` (10), dark-office tags only.
3. **Weekly:** rotate pool 2–3 fresh stills (same allowlist theme).
4. **Monitor LTX negatives** в логах (`logs/ltx_*.log`) — people/hands/faces/text/ui/bright rejects.
5. **Monthly:** log YOLO/OCR reject counts (reason tags) → trim pool / prompts.
6. Schedule: Schedule wrapper → `run.py --engine long` (gold defaults). Set `LTX_FACTORY`, `LTX_COMFY_PY`, `LTX_OUTPUT_DIR`, `LTX_FFMPEG` as needed.

---

## File map

| Path | Role |
|------|------|
| `config/video_assembler.yaml` | runtime config (code loads) |
| `config/gold_standard_v3.yaml` | canonical documented twin |
| `GOLD_STANDARD_v3.md` | this human doc |
| `shot_gate.py` | YOLO/CLIP/OCR/luma constants |
| `run_scheduled.ps1` | 11:00/18:00 wrapper |
| Env vars | `LTX_FACTORY`, `LTX_COMFY_PY`, `LTX_OUTPUT_DIR`, `LTX_FFMPEG` |

Backup prefix before edits: `*.bak.pre-gold-v3-YYYYMMDD-HHMM`
