"""Groq chat-completions client for grounded RAG responses."""

import os

import requests
from dotenv import load_dotenv


load_dotenv()


class GroqLLMClient:
    """Call Groq's OpenAI-compatible chat-completions endpoint."""

    endpoint = "https://api.groq.com/openai/v1/chat/completions"

    def __init__(self, model: str | None = None, timeout: int = 60):
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], api_key: str = "") -> str:
        key = api_key.strip() or os.getenv("GROQ_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "A Groq API key is required. Add GROQ_API_KEY to .env or enter it "
                "in the app sidebar."
            )

        response = requests.post(
            self.endpoint,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Groq returned an unexpected response format.") from error

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Groq returned an empty answer.")
        return content.strip()