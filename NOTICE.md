# Авторские права и сторонние компоненты

**Copyright (c) 2026 Yevhen Potapov. Все права защищены.**

Весь исходный код, документация, оформление и графика в этом проекте созданы
Yevhen Potapov и защищены авторским правом. Копирование, изменение и
использование допускаются только в рамках соответствующей лицензии
(см. `LICENSE` в каждом репозитории).

Ссылка на автора: [github.com/yevhenypupotapov-code](https://github.com/yevhenypupotapov-code)

---

## Сторонние компоненты

Проект **использует** перечисленное ниже, но **не распространяет** эти
компоненты в своём составе — они устанавливаются отдельно на компьютер
пользователя. Права принадлежат их авторам.

| Компонент | Назначение в проекте | Лицензия (у автора) |
|---|---|---|
| [Ollama](https://ollama.com) | локальный запуск языковой модели | MIT |
| [ComfyUI](https://github.com/comfyanonymous/ComfyUI) | движок генерации кадров и видео | GPL-3.0 |
| [LTX-Video](https://github.com/Lightricks/LTX-Video) (модели Lightricks) | модель генерации видео | собственная лицензия моделей Lightricks |
| [FFmpeg](https://ffmpeg.org) | монтаж, кодирование, звук | LGPL / GPL |
| [Python](https://www.python.org) | среда выполнения | PSF License |
| [Pillow](https://python-pillow.org) | работа с изображениями | MIT-CMU (HPND) |
| [NumPy](https://numpy.org) | вычисления | BSD-3-Clause |
| [MoviePy](https://zulko.github.io/moviepy/) | обработка видео | MIT |
| [imagehash](https://github.com/JohannesBuchner/imagehash) | сравнение кадров | BSD-2-Clause |
| [google-api-python-client](https://github.com/googleapis/google-api-python-client) | публикация на YouTube | Apache-2.0 |
| [React](https://react.dev) · [Vite](https://vitejs.dev) | веб-панель управления | MIT |
| Системный синтез речи Windows (SAPI) | озвучка | входит в состав Windows |

## Что проект **не** использует

В создании контента не применяются облачные генераторы: ни текстовые,
ни графические, ни голосовые. Текст считает локальная модель, кадры рисует
локальный ComfyUI, голос даёт системный синтез речи. Из интернета приходят
только тренды и загрузка готового ролика на YouTube.

## Товарные знаки

Названия сторонних продуктов принадлежат их владельцам и упомянуты
исключительно для указания факта использования.
