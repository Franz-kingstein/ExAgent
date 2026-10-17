"""Replaceable SLM providers for the rationale layer.

TRACE-XAI is not coupled to any model. The generator talks to the narrow
``RationaleProvider`` interface only:

    generate(system_prompt, user_prompt) -> str

Implementations:

- ``OllamaRationaleProvider``   — local model (default, e.g. qwen3.5:4b)
- ``OpenAICompatProvider``      — any OpenAI-compatible HTTP endpoint,
                                  isolated behind urllib (no extra deps);
                                  only reachable when explicitly configured
- ``EchoRationaleProvider``     — test double (returns the prompt; never used
                                  in production paths)

The generator itself has a deterministic non-LLM fallback, so the rationale
layer keeps working when no model is reachable.
"""

from __future__ import annotations

import os
import json
import urllib.request
from typing import Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """The provider failed to produce a usable response."""


@runtime_checkable
class RationaleProvider(Protocol):
    """Minimal interface: two prompts in, one text out."""

    name: str

    def generate(self, system_prompt: str, user_prompt: str) -> str: ...


class OllamaRationaleProvider:
    """Local Ollama model via the raw HTTP API (`/api/chat`).

    Uses urllib rather than langchain-ollama for two reasons:

    1. ``think: false`` — qwen3.5:4b otherwise burns its whole output budget
       on a hidden reasoning block (empty content, ``done_reason=length``).
       Disabling thinking also complies with the no-hidden-CoT research rule:
       the model works purely from the supplied evidence packet.
    2. The raw response exposes usage stats (eval_count, total_duration),
       which are kept in ``last_usage`` for the latency report.
    """

    name = "ollama"

    def __init__(self, model: str = "qwen3.5:4b", temperature: float = 0,
                 seed: int = 42, base_url: str = "http://localhost:11434",
                 timeout: float = 180.0, num_predict: int = 2048) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._options = {"temperature": temperature, "seed": seed,
                         "num_predict": num_predict}
        self.name = f"ollama/{model}"
        self.last_usage: dict = {}

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": False,
            "options": self._options,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/chat", data=payload,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as raw:
                body = json.loads(raw.read().decode("utf-8"))
        except Exception as exc:  # daemon down, model missing, ...
            raise ProviderError(f"ollama request failed: {exc}") from exc
        self.last_usage = {
            "eval_count": body.get("eval_count"),
            "prompt_eval_count": body.get("prompt_eval_count"),
            "total_duration_ns": body.get("total_duration"),
            "done_reason": body.get("done_reason"),
        }
        content = (body.get("message") or {}).get("content") or ""
        if not content.strip():
            raise ProviderError(
                f"ollama returned an empty response "
                f"(done_reason={body.get('done_reason')})"
            )
        return content


class OpenAICompatProvider:
    """Isolated external-provider interface (OpenAI-compatible /chat/completions).

    Never enabled by default. Requires explicit base_url (+ optional key env
    var). Kept dependency-free via urllib; this is the single choke point for
    sending data off-box — see docs/safety.md before enabling. All TRACE-XAI
    data is synthetic, but external calls are opt-in for reproducibility.
    """

    name = "openai-compat"

    def __init__(self, model: str, base_url: str, api_key_env: str = "TRACE_XAI_API_KEY",
                 timeout: float = 60.0) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key_env = api_key_env
        self._timeout = timeout
        self.name = f"openai-compat/{model}"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        key = os.environ.get(self._api_key_env)
        payload = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=payload, headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as raw:
                body = json.loads(raw.read().decode("utf-8"))
        except Exception as exc:
            raise ProviderError(f"openai-compat request failed: {exc}") from exc
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("provider returned an empty response")
        return content


class EchoRationaleProvider:
    """Test double: echoes the user prompt (never for production use)."""

    name = "echo-test"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return user_prompt
