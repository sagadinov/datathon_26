# Контроль кассовой дисциплины по видео — Datathon '26, кейс ForteBank

Локальный AI-сервис: загрузка видео с камеры кассы → вердикт, нарушения с таймкодом,
кадрами-доказательствами, описанием и ссылкой на норматив РК. Никаких облачных AI-API.

- Запуск и использование: [docs/HOWTO.md](docs/HOWTO.md)
- Архитектура: [docs/architecture.md](docs/architecture.md)
- Типы нарушений: [docs/violations.md](docs/violations.md)
- Нормативные источники (сверены с adilet.zan.kz): [docs/legal_sources.md](docs/legal_sources.md)
- Ограничения и известные ошибки: [docs/limitations.md](docs/limitations.md)

API (заголовок `X-API-Key`):

```bash
curl -X POST $URL/api/v1/analyze -H "X-API-Key: $KEY" -F "file=@video.mov"   # -> {"job_id": "..."}
curl $URL/api/v1/jobs/<job_id> -H "X-API-Key: $KEY"                          # -> status + result
```
