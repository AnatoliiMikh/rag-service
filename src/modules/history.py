# src/modules/history.py

import os
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv

load_dotenv()

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_DB = os.getenv("POSTGRES_DB", "chat_db")
POSTGRES_USER = os.getenv("POSTGRES_USER", "postgres")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
HISTORY_N = int(os.getenv("HISTORY_N", "10"))


class HistoryModule:
    def __init__(self):
        self._pool = None  # no connection attempt at startup

    def _get_pool(self):
        if self._pool is None:
            self._pool = psycopg2.pool.SimpleConnectionPool(
            minconn=1,
            maxconn=10,
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            dbname=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD,
            )
        return self._pool


    def get(self, chat_id: str, n: int = HISTORY_N) -> list[dict]:
        try:
            pool = self._get_pool()        # ← use the method, not self._pool directly
            conn = pool.getconn()
            try:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT role, content
                        FROM messages
                        WHERE chat_id = %s
                        ORDER BY created_at DESC
                        LIMIT %s
                    """, (chat_id, n))
                    rows = cur.fetchall()
                    return [
                        {"role": row[0], "content": row[1]}
                        for row in reversed(rows)
                    ]
            finally:
                pool.putconn(conn)         # ← use pool variable, not self._pool
        except Exception as e:
            print(f"[HistoryModule] PostgreSQL unavailable: {e}")
            return []