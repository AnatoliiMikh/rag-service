# src/pipeline.py

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.reranker_service import RerankerService
from services.bm25_service import BM25Service
from modules.history import HistoryModule
from modules.retrieval import RetrievalModule
from modules.context_builder import build_context


class RAGPipeline:
    def __init__(
        self,
        llm: LLMService,
        embedder: EmbeddingService,
        reranker: RerankerService,
        bm25: BM25Service,
        history: HistoryModule,
        retrieval: RetrievalModule,
    ):
        self._llm = llm
        self._embedder = embedder
        self._reranker = reranker
        self._bm25 = bm25
        self._history = history
        self._retrieval = retrieval

    def run(self, user_id: str, chat_id: str, message: str):
        """
        Orchestrates full RAG pipeline.
        Yields token strings for gRPC streaming.
        """
        history = self._history.get(chat_id)
        queries = self._llm.expand_query(message)

        vectors = self._embedder.embed(queries)
        dense_lists = self._retrieval.retrieve(vectors)
        bm25_lists = [self._bm25.search(q) for q in queries]

        chunks = self._reranker.rerank(message, dense_lists, bm25_lists)
        messages = build_context(message, history, chunks)

        yield from self._llm.generate(messages)