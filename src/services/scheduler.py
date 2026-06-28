# src/services/scheduler.py

import asyncio
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineRequest:
    user_id: str
    chat_id: str
    message: str
    result_queue: asyncio.Queue = field(default_factory=asyncio.Queue)


class Scheduler:
    def __init__(self, pipeline):
        self._pipeline = pipeline
        self._queue = asyncio.Queue()
        self._running = False

    async def submit(self, user_id: str, chat_id: str, message: str) -> asyncio.Queue:
        """
        Submits a request to the queue.
        Returns the result queue - caller awaits tokens from it.
        None sentinel signals end of stream.
        """
        request = PipelineRequest(
            user_id=user_id,
            chat_id=chat_id,
            message=message,
        )
        await self._queue.put(request)
        return request.result_queue

    async def run(self):
        """
        Processes requests sequentially.
        One request at a time - correct behavior under E. Process.
        """
        self._running = True
        print("[Scheduler] Running.")

        while self._running:
            request = await self._queue.get()

            try:
                for token in self._pipeline.run(
                    user_id=request.user_id,
                    chat_id=request.chat_id,
                    message=request.message,
                ):
                    await request.result_queue.put(token)

            except Exception as e:
                await request.result_queue.put(Exception(e))

            finally:
                # None signals end of stream to gRPC server
                await request.result_queue.put(None)
                self._queue.task_done()

    async def stop(self):
        self._running = False