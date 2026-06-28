# src/modules/context_builder.py

SYSTEM_PROMPT = """\
You are a precise and helpful university information assistant.
Your answers are based STRICTLY on the provided context chunks below.

Rules you must follow:
- Answer in the same language the user wrote in.
- If the context does not contain enough information to answer, say exactly:
  "I cannot find this information in the available documents."
- Never speculate or use knowledge outside the provided context.
- Always cite your sources at the end of your answer in this format:
  [Source: <source_file>, Page <page>]
- Be concise and direct. Do not repeat the question back.
"""


def build_context(
    message: str,
    history: list[dict],
    chunks: list[dict],
) -> list[dict]:
    """
    Assembles the final prompt as a list of messages.
    Combines system instructions, retrieved chunks with citations, chat history,
    and the current user message.
    Returns list[dict] ready to pass directly to client.chat.completions.create()
    """

    # Format retrieved chunks with source citations
    context_str = "\n\n".join(
        f"[Chunk {i+1} | Source: {c['source_file']} | Page {c['page']} | Score: {c['ce_score']:.3f}]\n{c['text']}"
        for i, c in enumerate(chunks)
    )

    # System message with injected context
    system_message = {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}\n\n### RETRIEVED CONTEXT\n{context_str}"
    }

    # Chat history as-is (already list of {"role": ..., "content": ...})
    history_messages = history

    # Current user message
    user_message = {
        "role": "user",
        "content": message,
    }

    return [system_message] + history_messages + [user_message]
