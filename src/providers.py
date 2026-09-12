"""Pluggable LLM backends with on-disk caching.

Both backends: deterministic where the API allows it, raw response logged
verbatim, and every call cached so re-runs are free and reproducible.

Determinism caveat, which goes in the paper: Claude Sonnet 5 does not accept
`temperature`/`top_p`/`top_k` (the API returns 400), so "temperature 0" cannot be
claimed for it. Extended thinking is disabled explicitly on both backends so that
reasoning is manipulated only through the prompt - otherwise C1 would silently
carry hidden reasoning and the C1/C2 contrast would be meaningless.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "cache"


def _load_env() -> None:
    """Read KEY=VALUE from .env (project root, then its parent). Never overwrites a real env var."""
    import os
    for path in (ROOT / ".env", ROOT.parent / ".env"):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_env()


class ProviderError(RuntimeError):
    """A call failed. Surfaced, never silently swallowed or retried into a different result."""


@dataclass
class Response:
    text: str
    cached: bool
    meta: dict = field(default_factory=dict)


class Provider(Protocol):
    name: str
    model: str

    def complete(self, system: str, user: str, seed: int) -> Response: ...


def _key(parts: dict) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


class _Cached:
    """Shared cache/logging. Subclasses implement _call()."""

    name: str
    model: str

    def _params(self) -> dict:
        """Everything that affects the output, so the cache cannot collide."""
        raise NotImplementedError

    def _call(self, system: str, user: str, seed: int) -> tuple[str, dict]:
        raise NotImplementedError

    def complete(self, system: str, user: str, seed: int) -> Response:
        parts = {"provider": self.name, "model": self.model, "system": system,
                 "user": user, "seed": seed, **self._params()}
        h = _key(parts)
        path = CACHE / self.name / f"{h}.json"
        if path.exists():
            rec = json.loads(path.read_text())
            return Response(rec["response"], cached=True, meta=rec.get("meta", {}))

        t0 = time.time()
        try:
            text, meta = self._call(system, user, seed)
        except Exception as e:  # surface, do not retry into a different result
            raise ProviderError(f"{self.name}/{self.model}: {type(e).__name__}: {e}") from e
        meta["latency_s"] = round(time.time() - t0, 3)
        # Provenance. Hosted model IDs can be silently re-pointed at a different
        # model (deepseek-v4-pro -> V4.1-Flash on 2026-09-14 04:00 UTC), so the
        # wall-clock time of the call is part of what identifies the model.
        meta["called_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"request": parts, "response": text, "meta": meta}, indent=2, ensure_ascii=False))
        return Response(text, cached=False, meta=meta)


class AnthropicProvider(_Cached):
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5", max_tokens: int = 2048):
        import anthropic
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def _params(self) -> dict:
        # No temperature: Sonnet 5 rejects sampling params with a 400.
        return {"max_tokens": self.max_tokens, "thinking": "disabled"}

    def _call(self, system: str, user: str, seed: int) -> tuple[str, dict]:
        r = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": user}],
        )
        if r.stop_reason == "refusal":
            raise ProviderError(f"refusal: {r.stop_details}")
        text = "".join(b.text for b in r.content if b.type == "text")
        return text, {"stop_reason": r.stop_reason,
                      "in_tokens": r.usage.input_tokens,
                      "out_tokens": r.usage.output_tokens}


class OllamaProvider(_Cached):
    name = "ollama"

    def __init__(self, model: str = "qwen3:8b", host: str = "http://localhost:11434",
                 num_predict: int = 2048):
        self.model = model
        self.host = host
        self.num_predict = num_predict

    def _params(self) -> dict:
        return {"temperature": 0, "num_predict": self.num_predict, "think": False}

    def _healthy(self) -> bool:
        import urllib.request
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=5)
            return True
        except Exception:
            return False

    def _call(self, system: str, user: str, seed: int) -> tuple[str, dict]:
        """Retry transport failures only.

        The local server can be killed by memory pressure mid-run; that is a
        transport failure, not a model failure. Re-issuing the identical request
        at temperature 0 with a fixed seed returns the identical result, so this
        is not "retrying into a different result" - it is resuming after a crash.
        A model-level failure still propagates on the first occurrence.
        """
        import urllib.error
        last = None
        for attempt in range(5):
            try:
                return self._generate(system, user, seed, attempt)
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last = e
                for _ in range(60):           # up to ~5 min for the server to return
                    time.sleep(5)
                    if self._healthy():
                        break
        raise ProviderError(
            f"{self.name}/{self.model}: transport failed after 5 attempts: {last}")

    def _generate(self, system: str, user: str, seed: int, attempt: int) -> tuple[str, dict]:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "system": system,
            "prompt": user,
            "stream": False,
            "think": False,          # qwen3 is a thinking model; keep C1 honest
            "options": {"temperature": 0, "seed": seed, "num_predict": self.num_predict},
        }).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read())
        return body["response"], {
            "done_reason": body.get("done_reason"),
            "in_tokens": body.get("prompt_eval_count"),
            "out_tokens": body.get("eval_count"),
            "transport_retries": attempt,
        }


class OpenAICompatProvider(_Cached):
    """DeepSeek, Zhipu GLM, and anything else exposing an OpenAI-compatible endpoint.

    These accept `temperature`, so unlike Sonnet 5 they can genuinely run at 0.
    `seed` is passed when the endpoint honours it; it is recorded in the cache key
    regardless so runs stay distinguishable.
    """

    ENDPOINTS = {
        "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
        "glm": ("https://open.bigmodel.cn/api/paas/v4", "ZHIPU_API_KEY"),
        "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    }

    # How each vendor turns reasoning OFF. Verified by probe, not assumed: DeepSeek
    # silently IGNORES OpenRouter's `reasoning: {enabled: false}` and keeps emitting
    # reasoning_content, which would destroy the C1-vs-C2 contrast without any error.
    NO_REASONING = {
        "deepseek": {"reasoning_effort": "none"},
        "openrouter": {"reasoning": {"enabled": False}},
        "glm": {"thinking": {"type": "disabled"}},
    }

    def __init__(self, vendor: str, model: str, max_tokens: int = 2048,
                 pin_provider: str | None = None):
        import os
        if vendor not in self.ENDPOINTS:
            raise ValueError(f"unknown vendor {vendor!r}; known: {list(self.ENDPOINTS)}")
        base, env = self.ENDPOINTS[vendor]
        self.name = vendor
        self.model = model
        self.base = os.environ.get(f"{vendor.upper()}_BASE_URL", base)
        self.max_tokens = max_tokens
        # OpenRouter picks a downstream host per request; hosts differ in quantisation
        # and kernels, so an unpinned run is not reproducible. Pin it.
        self.pin_provider = pin_provider or os.environ.get("OPENROUTER_PIN_PROVIDER")
        self._key_env = env
        self._api_key = os.environ.get(env)
        if not self._api_key:
            raise ProviderError(f"{env} is not set")

    def _params(self) -> dict:
        return {"temperature": 0, "max_tokens": self.max_tokens,
                "no_reasoning": self.NO_REASONING[self.name],
                "pin_provider": self.pin_provider}

    def _call(self, system: str, user: str, seed: int) -> tuple[str, dict]:
        import urllib.request
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "seed": seed,
            "stream": False,
            # Same reason thinking is disabled on the other backends: hidden reasoning
            # in C1 would destroy the C1-vs-C2 contrast. Vendor-specific - see NO_REASONING.
            **self.NO_REASONING[self.name],
            **({"provider": {"order": [self.pin_provider], "allow_fallbacks": False}}
               if self.pin_provider else {}),
        }).encode()
        req = urllib.request.Request(
            f"{self.base}/chat/completions", data=payload,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self._api_key}"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            body = json.loads(resp.read())
        if "choices" not in body:
            raise ProviderError(f"no choices in response: {str(body)[:300]}")
        choice = body["choices"][0]
        usage = body.get("usage", {})
        return choice["message"]["content"], {
            "finish_reason": choice.get("finish_reason"),
            "in_tokens": usage.get("prompt_tokens"),
            "out_tokens": usage.get("completion_tokens"),
            "served_by": body.get("provider"),   # which host OpenRouter actually used
            # Non-empty means reasoning leaked through despite NO_REASONING - the run
            # is then not a valid C1/C2 contrast and must be discarded.
            "reasoning_leak": len(choice["message"].get("reasoning_content") or ""),
        }


def build(spec: str) -> Provider:
    """e.g. `anthropic:claude-sonnet-5`, `ollama:qwen3:8b`,
    `deepseek:<model>`, `glm:<model>`."""
    kind, _, model = spec.partition(":")
    if kind == "anthropic":
        return AnthropicProvider(model=model)
    if kind == "ollama":
        return OllamaProvider(model=model)
    if kind in OpenAICompatProvider.ENDPOINTS:
        return OpenAICompatProvider(vendor=kind, model=model)
    raise ValueError(f"unknown provider spec: {spec!r}")
