# test_pipeline.py (put this in repo root)

import sys
sys.path.insert(0, 'src')

from modules.query_expansion import expand_query
from modules.embedding import embed_queries
from modules.reranker import rerank
from modules.context_builder import build_context
from modules.llm import generate

# Test message
message = "What are the English language requirements for international students?"

print("=== Step 1: Query Expansion ===")
queries = expand_query(message)
print(queries)

print("\n=== Step 2: Embedding ===")
vectors = embed_queries(queries)
print(f"Embedded {len(vectors)} queries, dense dim: {len(vectors[0].dense)}")

print("\n=== Step 3: LLM Generation (direct, no retrieval yet) ===")
messages = build_context(message, [], [{"text": "International students need IELTS 6.5.", "source_file": "test.pdf", "page": 1, "ce_score": 0.95}])
answer = generate(messages)
print(answer)