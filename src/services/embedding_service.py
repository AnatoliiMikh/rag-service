# src/services/embedding_service.py

import os
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")


@dataclass
class QueryVectors:
    query: str
    dense: list[float]


class EmbeddingService:
    def __init__(self):
        print("[EmbeddingService] Loading BGE-M3...")
        self._model = SentenceTransformer(EMBEDDING_MODEL_PATH, device="cuda")
        print("[EmbeddingService] Ready.")

    def embed(self, queries: list[str]) -> list[QueryVectors]:
        """
        Encodes all queries in one batch call.
        Returns dense vectors per query.
        """
        embeddings = self._model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [
            QueryVectors(query=query, dense=embeddings[i].tolist())
            for i, query in enumerate(queries)
        ]