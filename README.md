# LTX → YouTube на YEVGEN (RTX 3070 Laptop 8GB)

ComfyUI у тебя уже живой: `http://127.0.0.1:8188`, выход смотрим в `D:\Tutorial\Output` и `ComfyUI\output`.

## Что такое LTX-Video

Семейство открытых DiT-моделей Lightricks (Facetune / Videoleap): text-to-video и image-to-video.

На этой машине уже стоит **локальный LTX-2.5 22B distilled INT8** (transformer + Gemma 4 + video/audio VAE). С 17 августа у тебя в `ComfyUI\output\video` лежат `LTX-2.5_i2v_*.mp4` и `LTX_2.5_t2v_*.mp4` — то есть 2.5 на 8GB у тебя уже оживает, просто медленно и с offload.

Для автопайплайна надёжнее стартовать с **LTXV 2B distilled FP8**: один файл ~4.5 ГБ, изначально целился в 8GB, 768×512, 49 кадров (~2 сек), 8 шагов.

Не путай шаблон ComfyUI `api_ltx2_5_*` — это **облако Lightricks** (`LtxApi25TextToVideo`), не твой GPU. Локальный 2.5 шаблон: `Templates → Video → Text to Video (LTX-2.5)`.

## Что качать (2B, для этого скрипта)

| Файл | Куда | URL |
|---|---|---|
| `ltxv-2b-0.9.8-distilled-fp8.safetensors` (~4.46 GB) | `D:\ComfyUI\ModelHub\models\checkpoints` | [Hugging Face](https://huggingface.co/Lightricks/LTX-Video/resolve/main/ltxv-2b-0.9.8-distilled-fp8.safetensors) |
| `t5xxl_fp8_e4m3fn.safetensors` (~4.9 GB) | `D:\ComfyUI\ModelHub\models\text_encoders` | [Hugging Face](https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors) |

Либо из этой папки:

```powershell
powershell -ExecutionPolicy Bypass -File .\download_ltxv_2b.ps1
```

2.5 у тебя **уже скачан**, качать ещё раз не надо:

- `diffusion_models\ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors`
- `text_encoders\gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors`
- `vae\ltx-2.5-video-vae-bf16.safetensors` + `ltx-2.5-audio-vae-bf16.safetensors`
- `latent_upscale_models\ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors`

Дубль `ltx-2.5-22b-... (1).safetensors` можно удалить — это копия на 20 ГБ.

Настройки 2B на 8GB: **768×512, 49 кадров (8n+1), 8 steps, cfg=1, euler**. Промпт длинный, на английском.

## Workflow

Готовые API-графы (то, что жрёт `/prompt`):

- `workflows/ltxv_2b_t2v_api.json`
- `workflows/ltxv_2b_i2v_api.json`

Для локального **LTX-2.5**: в ComfyUI открой шаблон Text to Video (LTX-2.5) → **File → Export (API)** → сохрани как `workflows/ltx25_t2v_api.json`. Потом:

```text
python generate_and_upload.py --workflow workflows/ltx25_t2v_api.json --prompt "..." --no-upload
```

На 8GB в 2.5 не ставь 1280×720 из шаблона. ResolutionSelector: **0.2–0.4 MP** (608×352 … 864×480), 5 секунд, `prompt_enhance=false`.

## Скрипт

```powershell
cd D:\Tutorial\ltx-youtube
copy config.example.json config.json
D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt

# только генерация, без YouTube
D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe generate_and_upload.py `
  --mode t2v --no-upload `
  --prompt "A cinematic close-up of a red fox walking through fresh snow at golden hour, steam rising from its breath, slow camera pan, highly detailed fur, natural lighting. No text."
```

I2V:

```powershell
...\python.exe generate_and_upload.py --mode i2v --image C:\path\frame.png --no-upload --prompt "The fox turns its head toward the camera, snow drifting, cinematic lighting."
```

ComfyUI должен быть запущен. Скрипт ставит промпт в очередь, ждёт mp4, печатает путь.

## YouTube

1. [Google Cloud Console](https://console.cloud.google.com/) → проект → Enable **YouTube Data API v3**.
2. OAuth client type **Desktop app** → скачай JSON → положи как `client_secrets.json` сюда.
3. Первый запуск откроет браузер (один раз), токен сохранится в `youtube_token.json`.
4. По умолчанию заливка **private**. Public/массовая автозаливка — это ToS YouTube, спам и дубли банят.

```powershell
...\python.exe generate_and_upload.py --mode t2v --prompt "..." --title "Fox test" --privacy private
```

## Чего ComfyUI сам не сделает

Уникальные заголовки, SEO, обложка, расписание Shorts. Это следующий слой (локальная LLM / отдельный скрипт). Сначала один стабильный mp4 в `output`, потом автопубликация.


## Один проект с твоим video_factory

Старый завод не удалял. AUTOPILOT по расписанию живёт в `C:\Users\yevhe\.openclaw\workspace`.
LTX подключён к тому же `token.pickle` и `topics_history.json`.

Единая точка входа:

```powershell
cd D:\Tutorial\ltx-youtube
# LTX-клип, без заливки
D:\ComfyUI\ComfyUI_windows_portable\python_embeded\python.exe run.py --engine ltx --no-upload --prompt "A red fox in snow, cinematic, no text"

# тема + сценарий с завода, картинка из LTX, заливка тем же YouTube-токеном
...\python.exe run.py --engine hybrid --topic "Local LLMs vs cloud" --lang en --privacy private

# старый 16:9 слайд-завод
python run.py --engine slides --lang ru --minutes 6 --no-upload
```

`--privacy` по умолчанию private, чтобы не пересечься с AUTOPILOT, который уже льёт public.
