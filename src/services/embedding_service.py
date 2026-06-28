# src/services/embedding_service.py

import os
from dataclasses import dataclass
from FlagEmbedding import BGEM3FlagModel
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", "BAAI/bge-m3")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))


@dataclass
class QueryVectors:
    query: str
    dense: list[float]
    sparse: dict[int, float]   # token_id -> weight


class EmbeddingService:
    def __init__(self):
        print("[EmbeddingService] Loading BGE-M3...")
        self._model = BGEM3FlagModel(
            EMBEDDING_MODEL_PATH,
            use_fp16=True,
            device="cuda",
        )
        print("[EmbeddingService] Ready.")

    def embed(self, queries: list[str]) -> list[QueryVectors]:
        """
        Encodes all queries in one batch call.
        Returns dense + sparse vectors per query.
        """
        output = self._model.encode(
            queries,
            batch_size=EMBEDDING_BATCH_SIZE,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        return [
            QueryVectors(
                query=query,
                dense=output["dense_vecs"][i].tolist(),
                sparse=output["lexical_weights"][i],
            )
            for i, query in enumerate(queries)
        ]