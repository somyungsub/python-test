from __future__ import annotations

import os

from dotenv import load_dotenv
from fastapi import FastAPI


# Load .env for local dev; in containers prefer real env vars.
load_dotenv()

app = FastAPI(title="test-api")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    return {"message": "hello", "openai_api_key_present": api_key_present}
