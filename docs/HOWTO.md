# HOWTO — установка, запуск и проверка сервиса

Сервис: FastAPI + фоновая очередь, пайплайн запускается отдельным процессом на каждое видео.
Рабочая директория везде ниже — корень репозитория (`/shared/home/sagyndyk.sagadinov/datathon`).

## 1. Установка

```bash
bash scripts/setup_env.sh        # .venv (Python 3.11 через uv), torch CPU, requirements.txt
bash scripts/download_models.sh  # веса -> models/hf (единственное место, где есть сеть)
cp .env.example .env             # заполнить API_KEY (и NGROK_DOMAIN для туннеля)
```

`ffmpeg` в PATH не нужен: пайплайн и API берут статический бинарник из пакета `imageio-ffmpeg`
(либо `FFMPEG_BIN`, либо `module load FFmpeg/7.0.2-GCCcore-13.3.0`).

## 2. Конфигурация (`.env`, не коммитится)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `API_KEY` | — (обязательна) | ключ для заголовка `X-API-Key` |
| `PORT` | `8000` | порт uvicorn (`scripts/serve.sh`) |
| `HOST` | `127.0.0.1` | адрес uvicorn; туннель ходит локально |
| `WORKERS` | `2` | сколько видео обрабатывается параллельно |
| `TORCH_THREADS` | `32` | CPU-потоков на один пайплайн |
| `TIME_BUDGET_SEC` | `170` | бюджет пайплайна на видео; жёсткий kill через +60 с |
| `MAX_UPLOAD_MB` | `500` | лимит размера файла (413) |
| `JOBS_DIR` | `<repo>/jobs` | каталог заданий |
| `NGROK_DOMAIN` | — | статический домен ngrok для `scripts/tunnel.sh` |
| `PIPELINE_DEVICE` | `cpu` | `--device` пайплайна (`cuda`, когда есть GPU) |
| `PIPELINE_VLM` | `0` | `1` — не передавать `--no-vlm` |
| `API_LOG_STDERR` | `0` | `1` — дублировать лог API в stderr (Docker) |

Переменные окружения имеют приоритет над `.env`: `PORT=18800 bash scripts/serve.sh`.

## 3. Запуск API

```bash
bash scripts/serve.sh                          # foreground, весь вывод -> logs/api.log
nohup bash scripts/serve.sh >/dev/null 2>&1 &  # в фоне (или tmux/screen)
curl -s http://127.0.0.1:8000/health           # {"status":"ok","queue":0,"workers":2,"running":0}
tail -f logs/api.log                           # тайминги заданий, ошибки
pkill -f "uvicorn src.api.main:app"            # остановить
```

Ровно **один** процесс uvicorn (очередь в памяти процесса, `--workers N` недопустим).
После перезапуска задания со статусом `processing` ставятся в очередь заново, уже готовый
`result.json` при этом не пересчитывается.

## 4. Публичный URL (ngrok)

```bash
bash scripts/tunnel.sh            # читает NGROK_DOMAIN и PORT из .env -> https://$NGROK_DOMAIN
bash scripts/tunnel.sh 18800      # другой локальный порт
```

Токен ngrok лежит в `~/.config/ngrok/ngrok.yml`, в репозиторий не попадает. Проверка извне
(телефон, внешняя машина): `bash scripts/api_smoke.sh https://<домен> <ключ>`.
На бесплатном домене `*.ngrok-free.dev` браузер один раз показывает страницу-предупреждение
ngrok («Visit Site»); curl и другие API-клиенты её не видят, на жюри это не влияет.

## 5. Smoke-тест

```bash
bash scripts/api_smoke.sh                                  # локально, ключ из .env
bash scripts/api_smoke.sh http://127.0.0.1:18800 "$API_KEY"
VIDEO=data/open/videos/X-20.mov bash scripts/api_smoke.sh   # другой ролик
```

Проверяет: `/health`; 401 без ключа и с неверным ключом (оба эндпоинта); 404 на неизвестный
job; 400 без поля `file`; загрузка ролика -> `job_id` -> опрос до `done` (печатает вердикт,
число нарушений, время); битый файл (`echo junk > bad.mov`) -> `failed`. Код возврата 0 = всё ок.

## 6. API (по ТЗ)

```bash
URL=http://127.0.0.1:8000; KEY=...
curl -X POST $URL/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@data/open/videos/X-1.mov"
# {"job_id":"3f2a9c1e"}
curl $URL/api/v1/jobs/3f2a9c1e -H "X-API-Key: $KEY"
# {"job_id":"3f2a9c1e","status":"processing"}
# {"job_id":"3f2a9c1e","status":"done","verdict":"violation","violations":[{"type":...,
#   "timecode_sec":44.2,"description":...,"evidence_frames":[{"timestamp_sec":44.2,
#   "image_base64":"/9j/..."}],"rule":{...}}]}
# {"job_id":"3f2a9c1e","status":"failed","error":"Не удалось декодировать видео: ..."}
```

| Код | Когда |
|---|---|
| 401 `{"detail":"invalid api key"}` | нет заголовка `X-API-Key` или ключ неверный |
| 400 | нет multipart-поля `file`, пустой файл, нечитаемый multipart |
| 413 | файл больше `MAX_UPLOAD_MB` |
| 404 `{"detail":"job not found"}` | неизвестный или некорректный `job_id` |

Принимаются `.mov` и `.mp4` (контейнер определяется по содержимому). Поле `rule` и другие
дополнительные поля нарушения передаются из пайплайна как есть; при `done` после сбоя
добавляется поле `note`. Swagger: `GET /docs` (без ключа). Статика UI, если собрана в
`src/ui/static/`, отдаётся с `/`. CORS открыт для всех источников.

## 7. Жизненный цикл задания

```
jobs/<job_id>/input.mov                 загруженный файл
jobs/<job_id>/status.json               статус, тайминги, ошибка (атомарная запись)
jobs/<job_id>/pipeline.log              stdout/stderr пайплайна
jobs/<job_id>/result.json               результат пайплайна (result.json.partial.json — промежуточный)
```

Воркер: проверка декодируемости (ffmpeg, один кадр) -> `python -m src.pipeline.run input.mov
--out result.json --device cpu --no-vlm --time-budget $TIME_BUDGET_SEC` с `TORCH_THREADS` ->
жёсткий таймаут `TIME_BUDGET_SEC + 60` с. Исходы:

| Ситуация | Статус |
|---|---|
| есть `result.json` | `done` |
| только `result.json.partial.json` | `done` с частичным результатом |
| файл не видео / «cannot open video» | `failed` + `error` |
| падение или таймаут без результата | `done`, `no_violation`, `[]`, `"note":"pipeline error, see log"` |

Жюри ждёт 5 минут на видео: при `WORKERS=2` и бюджете 170 с запас есть, если ролики не
приходят пачками быстрее, чем обрабатываются (`/health` показывает `queue` и `running`).

## 8. Пайплайн из командной строки

```bash
.venv/bin/python -m src.pipeline.run data/open/videos/X-1.mov --out out.json --device cpu --no-vlm
.venv/bin/python -m src.pipeline.run video.mov --out out.json --time-budget 170 --keep-debug
TORCH_THREADS=16 .venv/bin/python -m src.pipeline.run video.mov --out out.json
```

Код возврата 0 и файл `out.json` по схеме API; ненулевой код и текст ошибки в stderr при сбое.

## 9. Docker (НЕ ПРОВЕРЕНО: на кластере нет Docker)

```bash
docker build -f docker/Dockerfile -t datathon-cashier-api .     # из корня репозитория
cd docker && docker compose up -d --build                        # читает ../.env, порт API_PORT (8000)
docker compose logs -f api
```

Образ `python:3.11-slim` + torch CPU (`download.pytorch.org/whl/cpu`) + `requirements.txt`,
копируются `src/ eval/ scripts/`; `models/` монтируется read-only, `jobs/`, `logs/`, `cache/` —
томами; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Перед сборкой добавьте в корень
`.dockerignore` (`.venv data models logs jobs cache .git`), иначе контекст сборки будет огромным.
Внутри контейнера TORCH_THREADS задайте по числу доступных ядер.

## 10. Тесты API

```bash
.venv/bin/python -m pytest tests/test_api.py -q    # ~1 мин, синтетическое видео, реальный subprocess
```

## 11. Частые проблемы

- `API_KEY is not set` при старте — нет `.env` или пустой ключ.
- `done` с `note: pipeline error, see log` — смотреть `jobs/<id>/pipeline.log` и `logs/api.log`.
- Порт занят — `PORT=18800 bash scripts/serve.sh` и `bash scripts/tunnel.sh 18800`.
- Медленно — уменьшить `WORKERS` до 1 при малом числе ядер, проверить `TORCH_THREADS`.
