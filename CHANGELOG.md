# Changelog

All notable changes to the LTX YouTube Gold Standard package.

## [3.0.0] — 2026-09-05

### Added
- Gold Standard v3.0 PASS config (`config/video_assembler.yaml`, `config/gold_standard_v3.yaml`).
- Final assembler (`video_assembler.py`): hold ≤5.0s, phash near-dup, global reuse bans, sliding-window `super_category` diversity, hybrid ~70% stills / ~30% LTX with same-category still fallback.
- Final gates (`shot_gate.py`): YOLO person/face conf 0.15, CLIP desk/tech allowlist min_sim 0.22, OCR text-area ≤5%, dark luma + Laplacian insurance.
- Photoreal desk still helper (`photo_fallback.py`) — refuses geometric PIL stubs.
- Factory entrypoint (`run.py`) with env-based paths (`LTX_FACTORY`, `LTX_COMFY_PY`, `LTX_OUTPUT_DIR`, `LTX_FFMPEG`).
- CI workflow: Python syntax check + YAML parse of gold configs.
- MIT `LICENSE`.

### Notes
- PASS reference (unlisted): https://youtu.be/58K_BqdM-_E
- Media pools, OAuth tokens, and machine-local secrets are **not** published.
