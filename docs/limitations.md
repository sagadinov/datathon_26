# Known limitations

_TODO: FP/FN from eval, camera angle sensitivity, no-GPU degradation._

## Проверка устойчивости

Критерий «Устойчивость» (10 баллов) жюри проверяет на новых роликах, финальный — под другим
углом. Чтобы не ждать тестовых видео, прогоняем пайплайн на искажённых копиях открытого
датасета (CLAUDE.md, раздел 7). Копии делает `scripts/make_augmented.py` только средствами
ffmpeg (бинарник из `imageio-ffmpeg`, как в пайплайне; `module load` не нужен):

| вариант | что имитирует | фильтр ffmpeg |
|---|---|---|
| `hflip` | зеркальная расстановка: сейф слева, дверь справа | `hflip` |
| `rot_p10` / `rot_m10` | камера наклонена на ±10°; размер кадра сохранён, углы чёрные | `rotate=±10*PI/180` |
| `small` | хуже разрешение: длинная сторона 480 px (пропорции и чётность размеров сохранены) | `scale` |

Все копии: 30 fps, H.264 (libx264, crf 23, preset veryfast, yuv420p), без звука, те же имена
файлов в `data/aug/<вариант>/videos/` (матчинг по номеру N работает как для оригиналов).
Скрипт идемпотентен: готовые файлы не перекодируются, а только перепроверяются; запись идёт
во временный `.part`, поэтому прерванный запуск не оставляет битых файлов. Длительность
каждой копии сверяется с оригиналом (±0,2 с) через ffprobe (`$FFPROBE_BIN`, PATH, каталог
модуля `FFmpeg/7.0.2-GCCcore-13.3.0`; без ffprobe — из вывода `ffmpeg -i`). Таблица по файлам
и сводка по вариантам печатаются в консоль, числа сохраняются в `data/aug/manifest.json`,
лог — `logs/make_augmented.log`. Код возврата 1, если хоть один файл не собрался или не
прошёл проверку длительности.

```bash
# все 42 ролика x 4 варианта: 6 процессов ffmpeg по 2 потока, ~100 с на логин-ноде, 113 МБ
.venv/bin/python scripts/make_augmented.py --procs 6
# часть роликов / вариантов, пересборка, просмотр команд без запуска
.venv/bin/python scripts/make_augmented.py --only 4,20,46 --variants hflip,small
.venv/bin/python scripts/make_augmented.py --only 4 --force
.venv/bin/python scripts/make_augmented.py --only 20 --dry-run

# прогон оценки на варианте: разметка та же (open_labels.xlsx), видео берутся из --videos-dir
.venv/bin/python -m eval.evaluate --no-vlm --tag base
.venv/bin/python -m eval.evaluate --no-vlm --videos-dir data/aug/hflip/videos   --tag hflip
.venv/bin/python -m eval.evaluate --no-vlm --videos-dir data/aug/rot_p10/videos --tag rot_p10
.venv/bin/python -m eval.evaluate --no-vlm --videos-dir data/aug/rot_m10/videos --tag rot_m10
.venv/bin/python -m eval.evaluate --no-vlm --videos-dir data/aug/small/videos   --tag small
```

Как читать: сравниваем `summary` из `eval/results/*.json` базового прогона и каждого варианта
(`config.videos_dir` и `config.tag` показывают, на чём считалось). `recall_violation`,
`hit_3s` / `hit_5s` и `fp_on_clean` не должны проседать. Если проседает только `hflip` —
правило привязано к стороне кадра («сейф справа»); только `rot_*` — зоны слишком узкие или
опираются на горизонтальные линии; только `small` — пороги детектора завязаны на разрешение.

Ограничения проверки:
- это синтетические искажения того же ракурса, они не заменяют съёмку с другой точки;
- все варианты, включая `hflip`, перекодированы (crf 23 вместо исходных ~1,6 Мбит/с, файлы
  в 2–3 раза меньше), поэтому часть просадки может давать само сжатие, а не геометрия;
- при повороте теряются углы кадра (чёрные треугольники), объекты у границ могут выпасть;
- разметка таймкодов та же, что у оригиналов — варианты не меняют длительность (проверено).
