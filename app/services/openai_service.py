from __future__ import annotations

import os
from typing import Sequence

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class OpenAIService:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set. Add it to your .env file.")
        self.client = OpenAI(api_key=api_key)
        self.chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
        self.embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    def embed_text(self, text: str) -> list[float]:
        response = self.client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def generate_message(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": "You create concise attendance motivation messages."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
        )
        return (response.choices[0].message.content or "Welcome back! Keep your streak strong today.").strip()

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self.client.embeddings.create(model=self.embedding_model, input=list(texts))
        return [item.embedding for item in response.data]
