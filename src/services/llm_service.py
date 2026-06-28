# src/services/llm_service.py

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
)
from threading import Thread
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL_PATH   = os.getenv("LLM_MODEL_PATH", "Qwen/Qwen3-4B-Instruct-2507-FP8")
LLM_MAX_TOKENS   = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE  = float(os.getenv("LLM_TEMPERATURE", "0.1"))

EXPANSION_PROMPT = """\
You are a search query rewriter for a university information retrieval system.

Given the user's question, generate exactly 3 alternative search queries.
Each query must:
- Be semantically diverse (not just paraphrases)
- Target a different aspect or perspective of the question
- Be self-contained and specific
- Be in the same language as the original question

Respond with ONLY a JSON array of 3 strings. No explanation, no markdown.

Example input: "What GPA do I need to apply?"
Example output: ["minimum GPA requirement for admission", "academic score threshold for university application", "grade point average eligibility criteria for enrollment"]

User question: {message}
"""


class LLMService:
    def __init__(self):
        print("[LLMService] Loading tokenizer...")
        self._tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH)

        print("[LLMService] Loading model into GPU...")
        self._model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL_PATH,
            torch_dtype="auto",
            device_map="cuda",
        )
        self._model.eval()

        # Prefix KV cache — stores KV states for static system prompt prefix
        # Computed once on first request, reused across all subsequent requests
        self._prefix_kv: tuple | None = None
        self._prefix_token_len: int = 0

        print("[LLMService] Ready.")

    # ------------------------------------------------------------------ #
    #  Query Expansion                                                     #
    # ------------------------------------------------------------------ #

    def expand_query(self, message: str) -> list[str]:
        """
        Sends message to Qwen3-4B with structured output prompt.
        Returns exactly 3 semantically diverse query variations.
        Falls back to [message x3] on malformed output.
        """
        prompt = EXPANSION_PROMPT.format(message=message)

        messages = [{"role": "user", "content": prompt}]

        tokenized = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
            tokenize=True,
        ).to("cuda")

        with torch.inference_mode():
            output_ids = self._model.generate(
                tokenized,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = output_ids[0][tokenized.shape[1]:]
        raw = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        try:
            queries = json.loads(raw)
            if isinstance(queries, list) and len(queries) == 3:
                return queries
        except json.JSONDecodeError:
            pass

        print(f"[LLMService] expand_query: malformed output, falling back.\nRaw: {raw}")
        return [message, message, message]

    # ------------------------------------------------------------------ #
    #  Answer Generation                                                   #
    # ------------------------------------------------------------------ #

    def generate(self, messages: list[dict]):
        """
        Takes assembled OpenAI-format message list from context_builder.
        Streams tokens via TextIteratorStreamer.
        Yields token strings one by one.
        Uses prefix KV cache for system prompt reuse.
        """
        # Apply chat template
        input_ids = self._tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")

        # Build past_key_values from prefix cache if available
        past_kv = self._get_prefix_kv(messages)

        # If prefix cache hit - skip prefix tokens in input
        if past_kv is not None:
            input_ids = input_ids[:, self._prefix_token_len:]

        streamer = TextIteratorStreamer(
            self._tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        generation_kwargs = dict(
            input_ids=input_ids,
            past_key_values=past_kv,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
            streamer=streamer,
        )

        # Run generation in background thread so we can yield from streamer
        thread = Thread(target=self._model.generate, kwargs=generation_kwargs)
        thread.start()

        for token in streamer:
            yield token

        thread.join()

    # ------------------------------------------------------------------ #
    #  Prefix KV Cache                                                     #
    # ------------------------------------------------------------------ #

    def _get_prefix_kv(self, messages: list[dict]) -> tuple | None:
        """
        Extracts static system prompt prefix (everything before retrieved chunks).
        Computes KV states once, reuses across requests.
        Returns past_key_values tuple or None if not applicable.
        """
        if not messages or messages[0]["role"] != "system":
            return None

        system_content = messages[0]["content"]

        # Static prefix is everything before the retrieved context
        # System prompt structure: "SYSTEM_PROMPT\n\n### RETRIEVED CONTEXT\n{chunks}"
        split_marker = "### RETRIEVED CONTEXT"
        if split_marker not in system_content:
            return None

        static_prefix = system_content.split(split_marker)[0]

        # Compute prefix KV once
        if self._prefix_kv is None:
            prefix_messages = [{"role": "system", "content": static_prefix}]
            prefix_ids = self._tokenizer.apply_chat_template(
                prefix_messages,
                return_tensors="pt",
                add_generation_prompt=False,
            ).to("cuda")

            with torch.inference_mode():
                prefix_output = self._model(
                    prefix_ids,
                    use_cache=True,
                )
                self._prefix_kv = prefix_output.past_key_values
                self._prefix_token_len = prefix_ids.shape[1]

            print(f"[LLMService] Prefix KV cache built: {self._prefix_token_len} tokens cached.")

        return self._prefix_kv