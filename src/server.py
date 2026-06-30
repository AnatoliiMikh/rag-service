# src/server.py

import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import grpc
import grpc.aio

import rag_service_pb2
import rag_service_pb2_grpc

from services.llm_service import LLMService
from services.embedding_service import EmbeddingService
from services.reranker_service import RerankerService
from services.scheduler import Scheduler
from services.bm25_service import BM25Service
from modules.history import HistoryModule
from modules.retrieval import RetrievalModule
from modules.context_builder import build_context
from pipeline import RAGPipeline

GRPC_PORT = os.getenv("GRPC_PORT", "50051")


class MessageServiceServicer(rag_service_pb2_grpc.MessageServiceServicer):

    def __init__(self, scheduler: Scheduler):
        self._scheduler = scheduler

    async def GenerateReply(self, request, context):
        """
        Receives NewMessageRequest from C# app.
        Submits to scheduler queue.
        Streams TokenChunk per token, then final Completion.
        Sends Error on exception.
        """
        conversation_id = request.conversation_id
        user_message = request.user_message

        try:
            result_queue = await self._scheduler.submit(
                user_id=str(conversation_id),
                chat_id=str(conversation_id),
                message=user_message,
            )

            full_text = []

            while True:
                token = await result_queue.get()

                # None sentinel — stream finished
                if token is None:
                    break

                # Exception from pipeline
                if isinstance(token, Exception):
                    yield rag_service_pb2.NewMessageChunkResponse(
                        error=rag_service_pb2.Error(message=str(token)),
                        conversation_id=conversation_id,
                    )
                    return

                full_text.append(token)

                # Stream each token to client
                yield rag_service_pb2.NewMessageChunkResponse(
                    token=rag_service_pb2.TokenChunk(text=token),
                    conversation_id=conversation_id,
                )

            # Send final completion with full assembled text
            yield rag_service_pb2.NewMessageChunkResponse(
                completion=rag_service_pb2.Completion(
                    full_text="".join(full_text)
                ),
                conversation_id=conversation_id,
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield rag_service_pb2.NewMessageChunkResponse(
                error=rag_service_pb2.Error(message=str(e)),
                conversation_id=conversation_id,
            )


async def main():
    print("[Server] Initializing services...")

    llm = LLMService()
    embedder = EmbeddingService()
    reranker = RerankerService()
    history = HistoryModule()
    retrieval = RetrievalModule()

    all_chunks = retrieval.load_all_chunks()
    bm25 = BM25Service(all_chunks)

    pipeline = RAGPipeline(
        llm=llm,
        embedder=embedder,
        reranker=reranker,
        bm25=bm25,
        history=history,
        retrieval=retrieval,
    )

    scheduler = Scheduler(pipeline)
    asyncio.create_task(scheduler.run())

    server = grpc.aio.server()
    rag_service_pb2_grpc.add_MessageServiceServicer_to_server(
        MessageServiceServicer(scheduler),
        server,
    )

    listen_addr = f"0.0.0.0:{GRPC_PORT}"
    server.add_insecure_port(listen_addr)
    await server.start()
    print(f"[Server] Listening on {listen_addr}")
    print("[Server] Ready.")

    async def shutdown():
        print("[Server] Shutting down...")
        await scheduler.stop()
        await server.stop(grace=5)

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(
        __import__("signal").SIGTERM,
        lambda: asyncio.create_task(shutdown()),
    )

    await server.wait_for_termination()

if __name__ == "__main__":
    asyncio.run(main())