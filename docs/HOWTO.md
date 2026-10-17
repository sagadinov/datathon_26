# How to run

```bash
bash scripts/setup_env.sh
bash scripts/download_models.sh
source .venv/bin/activate
python -m src.pipeline.run data/open/videos/X-1.mov --out out.json --device cpu --no-vlm
```
