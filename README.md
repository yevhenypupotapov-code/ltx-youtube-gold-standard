# LTX YouTube Gold Standard ⚙️ — ядро длинного завода

<p align="center">
  <a href="https://github.com/yevhenypupotapov-code/ltx-youtube-gold-standard/actions"><img src="https://img.shields.io/badge/ci-passing-2f6b4f?style=flat-square" alt="CI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?style=flat-square" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B8-%D0%BB%D0%BE%D0%BA%D0%B0%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5-b3402a?style=flat-square" alt="Local models only">
</p>

Обвязка длинного выпуска: берёт тему, собирает сценарий, генерирует кадры и LTXV-клипы,
склеивает всё в ролик со звуком и публикует. Работает полностью на локальном ПК.

---

## Состав

| Модуль | Роль |
|---|---|
| `run.py` | оркестратор: тема → сценарий → клипы → сборка → публикация |
| `video_assembler.py` | сборка таймлайна: удержание кадра, ротация сцен, аудиты качества |
| `shot_gate.py` | гейты кадра: яркость, резкость, «геометрия», OCR, пороги допуска |
| `add_audio.py` | озвучка и сведение со видео |
| `still_farm.py` | генерация свежих кадров для пула (бесконечная новизна) |
| `visual_identity.py` | цветовой облик выпуска: грейд из ротации без повторов |
| `desc_block.py` | описание под видео: суть, «как сделано», ссылки |

## Правила качества

- Удержание кадра не больше 5 секунд, сцена не повторяется подряд.
- Чёрные и «пустые» кадры отсекаются до публикации.
- Дедупликация: сравнение только со свежими выпусками и по доле совпавших
  ключевых кадров, а не по одной паре.
- Каждый выпуск получает новый визуальный облик (ротация грейдов).

## Быстрый старт

```bash
python run.py --engine long --lang ru --privacy public --segments 6
```

Требуется ComfyUI с LTXV на `127.0.0.1:8188` и Ollama на `127.0.0.1:11434`.

## Ссылки

- Приложение и панель: [contentforge](https://github.com/yevhenypupotapov-code/contentforge)
- Канал: [YEVHEN POTAPOV](https://www.youtube.com/@yevhenpotapov5956)

<p align="center"><sub>Локальные модели · Windows · 2026</sub></p>
