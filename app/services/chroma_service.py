from __future__ import annotations

from pathlib import Path

import chromadb


class ChromaService:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parents[2]
        persist_dir = base_dir / "data" / "chroma"
        persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name="attendance_memory")

    def get_recent_messages(self, user_id: int, limit: int = 5) -> list[str]:
        result = self.collection.get(where={"user_id": user_id}, include=["documents", "metadatas"])
        documents = result.get("documents", []) or []
        metadatas = result.get("metadatas", []) or []

        combined = list(zip(documents, metadatas))
        combined.sort(key=lambda item: item[1].get("timestamp", ""), reverse=True)
        return [doc for doc, _ in combined[:limit]]

    def save_message(self, user_id: int, timestamp: str, message: str, embedding: list[float]) -> None:
        item_id = f"{user_id}-{timestamp}"
        self.collection.add(
            ids=[item_id],
            embeddings=[embedding],
            documents=[message],
            metadatas=[{"user_id": user_id, "timestamp": timestamp}],
        )
