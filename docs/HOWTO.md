# HOWTO — установка, запуск и проверка сервиса

Сервис: FastAPI + фоновая очередь, пайплайн запускается отдельным процессом на каждое видео.
Рабочая директория везде ниже — корень репозитория (`/shared/home/sagyndyk.sagadinov/datathon`).

## Быстрый запуск на сервере (tmux)

```bash
bash scripts/start_service.sh   # API + ngrok-туннель в tmux-сессии fortebank, с авто-перезапуском
bash scripts/stop_service.sh    # остановить
bash scripts/api_smoke.sh https://<NGROK_DOMAIN> <API_KEY>   # проверка по ТЗ снаружи
```

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
| `LOGS_DIR` | `<repo>/logs` | `api.log` (API + access-лог uvicorn), `pipeline.log` |
| `NGROK_DOMAIN` | — | статический домен ngrok для `scripts/tunnel.sh` |
| `PIPELINE_DEVICE` | `cpu` | `--device` пайплайна (`cuda`, когда есть GPU) |
| `PIPELINE_VLM` | `0` | `1` — не передавать `--no-vlm` |
| `API_LOG_STDERR` | `0` | `1` — дублировать лог API в stderr (Docker) |

Переменные окружения имеют приоритет над `.env` и в `serve.sh`, и в `api_smoke.sh`:
`PORT=18800 bash scripts/serve.sh`, `API_KEY=... bash scripts/api_smoke.sh URL`.

## 3. Запуск, остановка, перезапуск

```bash
bash scripts/serve.sh                          # foreground, весь вывод -> logs/api.log
nohup bash scripts/serve.sh >/dev/null 2>&1 &  # в фоне (или tmux/screen)
curl -s http://127.0.0.1:8000/health           # {"status":"ok","queue":0,"workers":2,"running":0}
tail -f logs/api.log                           # тайминги заданий, ошибки
pkill -f "uvicorn src.api.main:app"            # остановить (SIGTERM, корректное завершение)
```

Ровно **один** процесс uvicorn (очередь в памяти процесса, `--workers N` недопустим).

`/health`: `queue` — заданий в очереди, `running` — пайплайнов в работе, `workers` — живых
рабочих потоков (`status` становится `degraded`, если их меньше `WORKERS`; упавший поток
пересоздаётся автоматически при следующем запросе).

Перезапуск не теряет задания:

- **Корректная остановка** (SIGTERM: `pkill`, Ctrl-C, `docker stop`, systemd): работающие
  пайплайны убиваются, их задания остаются в статусе `processing` и ставятся в очередь при
  следующем старте (в `logs/api.log` строка `api started ... requeued=N`). Клиент, который
  опрашивает `GET /api/v1/jobs/{id}`, увидит `processing`, затем результат.
- **Жёсткое убийство** (SIGKILL, OOM): пайплайн-процесс остаётся жить; после старта воркер
  находит его по `pipeline_pid` из `status.json` и **дожидается** его, а не запускает второй
  (в `status.json` будет `"adopted": true`).
- Уже готовый `result.json` при повторном запуске не пересчитывается. Счётчик `attempt` в
  `status.json`: после 3 неудачных попыток задание закрывается `done` с пустым результатом.

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
число нарушений, время, порядок полей); битый файл (`echo junk > bad.mov`) -> `failed` с текстом
ошибки без серверных путей. Код возврата 0 = всё ок.

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
# {"job_id":"3f2a9c1e","status":"failed","error":"Файл не содержит видеопотока"}
```

| Код | Когда |
|---|---|
| 401 `{"detail":"invalid api key"}` | нет заголовка `X-API-Key` или ключ неверный |
| 400 | нет multipart-поля `file`, пустой файл, нечитаемый multipart |
| 413 | файл больше `MAX_UPLOAD_MB` |
| 404 `{"detail":"job not found"}` | неизвестный или некорректный `job_id` |

Тексты `error` при `status: failed` (короткие, по-русски, без путей; подробности ffmpeg — в
`jobs/<id>/pipeline.log`): «Файл повреждён или не является видео: не удалось декодировать ни
одного кадра», «Файл не содержит видеопотока» (картинка или аудио под видом .mov), «Не удалось
проверить видеофайл: превышено время декодирования», «Загруженный файл не найден на сервере».

Принимаются `.mov` и `.mp4` (контейнер определяется по содержимому). Поле `rule` и другие
дополнительные поля нарушения передаются из пайплайна как есть; при `done` после сбоя
добавляется поле `note`. Swagger: `GET /docs` (без ключа). Статика UI, если собрана в
`src/ui/static/`, отдаётся с `/`. CORS открыт для всех источников.

## 7. Жизненный цикл задания

```
jobs/<job_id>/input.mov                 загруженный файл
jobs/<job_id>/status.json               статус, тайминги, attempt, pipeline_pid, ошибка (атомарная запись)
jobs/<job_id>/pipeline.log              stdout/stderr пайплайна (+ ошибки ffmpeg-проверки)
jobs/<job_id>/result.json               результат пайплайна (result.json.partial.json — промежуточный)
```

Воркер: проверка ffmpeg (два кадра при 2 fps, как семплирует пайплайн) -> `python -m
src.pipeline.run input.mov --out result.json --device cpu --no-vlm --time-budget $TIME_BUDGET_SEC`
с `TORCH_THREADS` в отдельной группе процессов -> жёсткий таймаут `TIME_BUDGET_SEC + 60` с. Исходы:

| Ситуация | Статус |
|---|---|
| есть `result.json` | `done` |
| только содержательный `result.json.partial.json` (после детекции) | `done` с частичным результатом |
| ffmpeg не открыл файл / «cannot open video» | `failed` + `error` |
| файл открывается, но кадров нет (картинка, аудио) | `failed`, «Файл не содержит видеопотока» |
| падение или таймаут без результата | `done`, `no_violation`, `[]`, `"note":"pipeline error, see log"` |
| остановка сервера во время обработки | остаётся `processing`, перезапуск ставит в очередь |

**Время ответа.** Жюри ждёт 5 минут на видео. Один пайплайн занимает не больше
`TIME_BUDGET_SEC + 60` с (худший случай 230 с при 170), обычно 10–130 с. При `WORKERS=2` третье
одновременно присланное видео ждёт очередь: худший случай 230 + 230 с > 5 мин. Если ролики
приходят пачками, либо `WORKERS=3` при свободных ядрах (3×`TORCH_THREADS` потоков), либо
`TIME_BUDGET_SEC=120` (запас: 180 + 180 с). `/health` показывает `queue` и `running`.

## 8. Пайплайн из командной строки

```bash
.venv/bin/python -m src.pipeline.run data/open/videos/X-1.mov --out out.json --device cpu --no-vlm
.venv/bin/python -m src.pipeline.run video.mov --out out.json --time-budget 170 --keep-debug
TORCH_THREADS=16 .venv/bin/python -m src.pipeline.run video.mov --out out.json
```

Код возврата 0 и файл `out.json` по схеме API; ненулевой код и текст ошибки в stderr при сбое.

## 9. Docker (НЕ ПРОВЕРЕНО: на кластере нет Docker)

```bash
printf '%s\n' .venv data models logs jobs cache .git .env '**/__pycache__' > .dockerignore  # один раз
docker build -f docker/Dockerfile -t datathon-cashier-api .     # из корня репозитория
cd docker && docker compose up -d --build                        # читает ../.env, порт API_PORT (8000)
docker compose logs -f api
docker compose stop                                              # SIGTERM: задания в работе -> processing -> re-queue при старте
```

Образ `python:3.11-slim` + torch CPU (`download.pytorch.org/whl/cpu`) + `requirements.txt`,
копируются `src/ eval/ scripts/`; `models/` (read-write: HF-кэш может создавать lock-файлы),
`jobs/`, `logs/`, `cache/` — томами; `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`. Без
`.dockerignore` контекст сборки включит `.venv`, данные и веса (десятки ГБ). Внутри контейнера
`TORCH_THREADS` задайте по числу доступных ядер. `requirements.txt` ставит и streamlit/pandas
(нужны eval и старому UI) — образ можно уменьшить, вынеся их в отдельный файл.

## 10. Тесты API

```bash
.venv/bin/python -m pytest tests/test_api.py -q    # ~2-3 мин, синтетическое видео, реальный subprocess
```

Покрывают: 401/404/400/413, битый файл и «картинка под видом .mov» -> `failed` с чистым текстом,
обычное видео -> `done`, падение пайплайна -> `done` + `note`, **остановка сервера во время
обработки** -> `processing` -> перезапуск -> `done`, **осиротевший пайплайн** после жёсткого
убийства дожидается, а не запускается второй раз, удаление `jobs/<id>` на ходу не убивает воркер.
`CACHE_DIR` и `LOGS_DIR` на время тестов уходят во временный каталог.

## 11. Частые проблемы

- `API_KEY is not set` при старте — нет `.env` или пустой ключ.
- `done` с `note: pipeline error, see log` — смотреть `jobs/<id>/pipeline.log` и `logs/api.log`.
- `failed` на нормальном видео — в `jobs/<id>/pipeline.log` строка `ffmpeg probe failed: ...`
  с сырым выводом ffmpeg (там же серверный путь, в API он не попадает).
- `/health` показывает `"status":"degraded"` — рабочий поток упал; в `logs/api.log` есть трассировка,
  поток уже пересоздан; если повторяется — проверить место на диске и права на `jobs/`.
- Порт занят — `PORT=18800 bash scripts/serve.sh` и `bash scripts/tunnel.sh 18800`.
- Медленно — уменьшить `WORKERS` до 1 при малом числе ядер, проверить `TORCH_THREADS`.
