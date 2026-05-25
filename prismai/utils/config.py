import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    MIMO_API_KEY = os.getenv("MIMO_API_KEY", "")
    MIMO_BASE_URL = os.getenv("MIMO_BASE_URL", "https://token-plan-sgp.xiaomimimo.com/v1")
    MIMO_MODEL = os.getenv("MIMO_MODEL", "mimo-v2.5-pro")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "50"))

    PROVIDER_CASCADE = [
        {"name": "mimo", "base_url": MIMO_BASE_URL, "api_key": MIMO_API_KEY, "model": MIMO_MODEL},
        {"name": "openrouter", "base_url": "https://openrouter.ai/api/v1", "api_key": OPENROUTER_API_KEY, "model": "deepseek/deepseek-chat-v3-0324"},
    ]
