# src/modules/retrieval.py

import os
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector, FusionQuery, Fusion
from dotenv import load_dotenv
from services.embedding_service import QueryVectors

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "university_docs")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))


class RetrievalModule:
    def __init__(self):
        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def retrieve(self, query_vectors: list[QueryVectors]) -> list[list[dict]]:
        """
        Hybrid search for each query vector.
        Dense + sparse fused via Qdrant's built-in RRF.
        Returns one ranked candidate list per query.
        """
        results = []

        for qv in query_vectors:
            hits = self._client.query_points(
                collection_name=QDRANT_COLLECTION,
                prefetch=[
                    {
                        "query": qv.dense,
                        "using": "dense",
                        "limit": TOP_K,
                    },
                    {
                        "query": SparseVector(
                            indices=list(qv.sparse.keys()),
                            values=list(qv.sparse.values()),
                        ),
                        "using": "sparse",
                        "limit": TOP_K,
                    },
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=TOP_K,
                with_payload=True,
            )

            results.append([
                {
                    "text": hit.payload.get("text", ""),
                    "source_file": hit.payload.get("source", ""),
                    "page": hit.payload.get("page", 0),
                    "score": hit.score,
                    "query": qv.query,
                }
                for hit in hits.points
            ])

        return results