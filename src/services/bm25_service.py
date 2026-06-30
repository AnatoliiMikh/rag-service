# src/services/bm25_service.py

from rank_bm25 import BM25Okapi


class BM25Service:
    def __init__(self, chunks: list[dict]):
        """
        Builds BM25 index from chunks loaded from Qdrant at startup.
        Pure CPU - no GPU, no torch, no CUDA context.
        """
        print(f"[BM25Service] Building index from {len(chunks)} chunks...")
        self._chunks = chunks
        tokenized = [chunk["text"].lower().split() for chunk in chunks]
        self._bm25 = BM25Okapi(tokenized)
        print("[BM25Service] Ready.")

    def search(self, query: str, k: int = 20) -> list[dict]:
        """
        Returns top k chunks by BM25 score for a single query.
        Skips zero-score chunks.
        """
        scores = self._bm25.get_scores(query.lower().split())
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:k]
        return [
            {**self._chunks[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]