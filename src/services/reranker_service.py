# src/services/reranker_service.py

import os
from FlagEmbedding import FlagReranker
from dotenv import load_dotenv

load_dotenv()

RERANKER_MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3")
TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))


class RerankerService:
    def __init__(self):
        print("[RerankerService] Loading BGE-Reranker...")
        self._reranker = FlagReranker(
            RERANKER_MODEL_PATH,
            use_fp16=True,
            device="cuda",
        )
        print("[RerankerService] Ready.")

    def rerank(self, query: str, candidate_lists: list[list[dict]]) -> list[dict]:
        """
        Step 1 - RRF: merges 3 ranked lists into deduplicated pool.
        Step 2 - Cross-encoder: scores (query, chunk) pairs.
        Returns top N chunks by cross-encoder score.
        """
        merged = self._rrf_merge(candidate_lists)

        if not merged:
            return []

        pairs = [[query, chunk["text"]] for chunk in merged]
        scores = self._reranker.compute_score(pairs, normalize=True)

        if not isinstance(scores, list):
            scores = [scores]

        for i, chunk in enumerate(merged):
            chunk["ce_score"] = scores[i]

        return sorted(merged, key=lambda c: c["ce_score"], reverse=True)[:TOP_N]

    def _rrf_merge(self, candidate_lists: list[list[dict]]) -> list[dict]:
        """
        Reciprocal Rank Fusion across 3 per-query candidate lists.
        Deduplicates by text. Boosts chunks appearing in multiple lists.
        Formula: score += 1 / (k + rank + 1)
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