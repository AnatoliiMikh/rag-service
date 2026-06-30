# src/services/reranker_service.py

import os
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))


class RerankerService:
    def __init__(self):
        print("[RerankerService] Loading BGE-Reranker...")
        self._reranker = CrossEncoder(
            RERANKER_MODEL_PATH,
            device="cuda",
        )
        print("[RerankerService] Ready.")

    def rerank(
        self,
        query: str,
        dense_lists: list[list[dict]],
        bm25_lists: list[list[dict]],
    ) -> list[dict]:
        """
        Step 1 - RRF: merges dense + BM25 lists into one pool.
        Step 2 - Cross-encoder: scores (query, chunk) pairs.
        Returns top N chunks by cross-encoder score.
        """
        merged = self._rrf_merge(dense_lists + bm25_lists)
        if not merged:
            return []

        pairs = [[query, chunk["text"]] for chunk in merged]
        scores = self._reranker.predict(pairs)

        for i, chunk in enumerate(merged):
            chunk["ce_score"] = float(scores[i])

        return sorted(merged, key=lambda c: c["ce_score"], reverse=True)[:TOP_N]

    def _rrf_merge(self, candidate_lists: list[list[dict]]) -> list[dict]:
        """
        Reciprocal Rank Fusion across all candidate lists.
        Deduplicates by text. Formula: score += 1 / (k + rank + 1)
        """
        rrf_scores: dict[str, float] = {}
        chunk_map: dict[str, dict] = {}

        for ranked_list in candidate_lists:
            for rank, chunk in enumerate(ranked_list):
                text = chunk["text"]
                rrf_scores[text] = rrf_scores.get(text, 0.0) + (1.0 / (RRF_K + rank + 1))
                if text not in chunk_map:
                    chunk_map[text] = chunk

        merged = []
        for text in sorted(rrf_scores, key=lambda t: rrf_scores[t], reverse=True):
            chunk = chunk_map[text].copy()
            chunk["rrf_score"] = rrf_scores[text]
            merged.append(chunk)

        return merged