# src/modules/retrieval.py

import os
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from services.embedding_service import QueryVectors

load_dotenv()

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "data_ds_bsc")
TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "20"))


class RetrievalModule:
    def __init__(self):
        self._client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    def load_all_chunks(self) -> list[dict]:
        """
        Fetches all chunks from Qdrant at startup.
        Used to build BM25 index.
        """
        results = []
        offset = None

        while True:
            response, offset = self._client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in response:
                results.append({
                    "text": point.payload.get("text", ""),
                    "source_file": point.payload.get("source", ""),
                    "page": point.payload.get("page", 0),
                    "id": point.id,
                })
            if offset is None:
                break

        print(f"[RetrievalModule] Loaded {len(results)} chunks from Qdrant.")
        return results

    def retrieve(self, query_vectors: list[QueryVectors]) -> list[list[dict]]:
        """
        Dense-only search in Qdrant for each expanded query.
        Returns one ranked candidate list per query.
        """
        try:
            results = []
            for qv in query_vectors:
                hits = self._client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=qv.dense,
                    using="",
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
                    for hit in hits
                ])
            return results
        except Exception as e:
            print(f"[RetrievalModule] Qdrant unavailable: {e}")
            return [[] for _ in query_vectors]