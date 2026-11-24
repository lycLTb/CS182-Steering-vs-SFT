"""All paths are relative to the yanchi/ folder, so it can live anywhere inside a shared repo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # .../yanchi
DATA = ROOT / "data"  # benchmark JSONL (git-ignored, from scripts/prepare_data.py) + data/ids (committed)
MODELS = ROOT / "models"  # local model dirs (git-ignored): base + every SFT/LoRA-merged model
OUTPUTS = ROOT / "outputs"  # generations / scores, one folder per evaluated model
BASE_DIR = MODELS / "qwen3.5-0.8b-base"  # created by scripts/export_base.py
