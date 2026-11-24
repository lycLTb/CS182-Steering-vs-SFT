"""DeepSeek judge API config. The key is NEVER stored in code (this folder is pushed to a shared GitHub repo):
set env DEEPSEEK_API_KEY, or put the key in yanchi/.deepseek_api_key (git-ignored)."""
import os

from core.paths import ROOT

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"  # = DeepSeek-V4.1-Flash; served by snapshot V4.1-Flash-1017 as of 2026-11 (track system_fingerprint)
KEY_FILE = ROOT / ".deepseek_api_key"


def api_key() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise SystemExit(f"No DeepSeek API key: export DEEPSEEK_API_KEY=... or write it to {KEY_FILE}")
    return key


def client(timeout=180):
    from openai import OpenAI
    return OpenAI(api_key=api_key(), base_url=BASE_URL, timeout=timeout)
