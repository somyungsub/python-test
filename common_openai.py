import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

MODEL = "gpt-4o-mini"

def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is missing. Check your .env file.")
    return api_key


def create_model(model: str = MODEL, temperature: float = 0) -> ChatOpenAI:
    return ChatOpenAI(model=model, api_key=get_api_key(), temperature=temperature)
