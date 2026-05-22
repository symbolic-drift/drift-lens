"""Unified AWS Bedrock client.

Wraps the Bedrock `invoke_model` API for the model families used in the
SymbolicDrift evaluation pipeline (Anthropic, OpenAI, DeepSeek, Qwen, Llama,
Mistral, Moonshot, MiniMax). Each provider has a slightly different request /
response schema; this client normalizes both.

Authentication uses the default AWS credential chain (env vars, ``AWS_PROFILE``,
instance profile, ...). No profile name is hardcoded.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Provider registry: each entry knows how to build a request body and parse a
# response body for one Bedrock model family.
# --------------------------------------------------------------------------- #

def _anthropic_body(system_prompt: str, user_prompt: str, *, max_tokens: int,
                    temperature: float, top_p: float | None,
                    thinking_budget: int | None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
        "temperature": temperature,
    }
    if top_p is not None:
        body["top_p"] = top_p
    if thinking_budget is not None:
        body["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
    return body


def _anthropic_parse(response: dict[str, Any], *, thinking: bool) -> str:
    content = response.get("content", []) or []
    if not content:
        return ""
    if thinking:
        # Skip the thinking block; return the first non-thinking text segment.
        for item in content:
            if item.get("type") == "text":
                return item.get("text", "")
    return content[0].get("text", "")


def _openai_chat_body(system_prompt: str, user_prompt: str, *, max_tokens: int,
                      temperature: float, top_p: float | None,
                      use_completion_tokens: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
    }
    body["max_completion_tokens" if use_completion_tokens else "max_tokens"] = max_tokens
    if top_p is not None:
        body["top_p"] = top_p
    return body


def _openai_chat_parse(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices:
        return choices[0].get("message", {}).get("content", "")
    if "content" in response:
        return response["content"]
    if "output" in response and isinstance(response["output"], dict):
        return response["output"].get("text", "")
    return response.get("text", "")


def _deepseek_r1_body(system_prompt: str, user_prompt: str, *, max_tokens: int,
                      temperature: float, top_p: float | None) -> dict[str, Any]:
    # DeepSeek R1 on Bedrock takes a single prompt string with chat tokens.
    prefix = f"{system_prompt}\n\n" if system_prompt else ""
    formatted = (
        "<｜begin▁of▁sentence｜><｜User｜>"
        f"{prefix}{user_prompt}"
        "<｜Assistant｜><think>\n"
    )
    return {
        "prompt": formatted,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.95 if top_p is None else top_p,
    }


def _deepseek_r1_parse(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if choices:
        return choices[0].get("text", "")
    return response.get("text", "")


def _llama_body(system_prompt: str, user_prompt: str, *, max_tokens: int,
                temperature: float, top_p: float | None) -> dict[str, Any]:
    formatted = (
        "<|begin_of_text|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{system_prompt} {user_prompt}\n"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )
    return {
        "prompt": formatted,
        "max_gen_len": max_tokens,
        "temperature": temperature,
        "top_p": 1.0 if top_p is None else top_p,
    }


def _llama_parse(response: dict[str, Any]) -> str:
    return response.get("generation", "")


@dataclass(frozen=True)
class ProviderSpec:
    build_body: Callable[..., dict[str, Any]]
    parse_response: Callable[..., str]
    needs_thinking_kwarg: bool = False
    body_kwargs: dict[str, Any] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderSpec] = {
    "anthropic": ProviderSpec(
        build_body=_anthropic_body,
        parse_response=_anthropic_parse,
        needs_thinking_kwarg=True,
    ),
    "openai": ProviderSpec(
        build_body=_openai_chat_body,
        parse_response=_openai_chat_parse,
        body_kwargs={"use_completion_tokens": True},
    ),
    "qwen": ProviderSpec(
        build_body=_openai_chat_body,
        parse_response=_openai_chat_parse,
        body_kwargs={"use_completion_tokens": False},
    ),
    "mistral": ProviderSpec(
        build_body=_openai_chat_body,
        parse_response=_openai_chat_parse,
        body_kwargs={"use_completion_tokens": False},
    ),
    "deepseek_v3": ProviderSpec(
        build_body=_openai_chat_body,
        parse_response=_openai_chat_parse,
        body_kwargs={"use_completion_tokens": False},
    ),
    "moonshot": ProviderSpec(
        build_body=_openai_chat_body,
        parse_response=_openai_chat_parse,
        body_kwargs={"use_completion_tokens": False},
    ),
    "minimax": ProviderSpec(
        build_body=_openai_chat_body,
        parse_response=_openai_chat_parse,
        body_kwargs={"use_completion_tokens": False},
    ),
    "deepseek_r1": ProviderSpec(
        build_body=_deepseek_r1_body,
        parse_response=_deepseek_r1_parse,
    ),
    "llama": ProviderSpec(
        build_body=_llama_body,
        parse_response=_llama_parse,
    ),
}


def infer_provider(model_id: str) -> str:
    """Best-effort mapping from a Bedrock model ID prefix to a provider key."""
    head = model_id.split(".")[0].lower()
    # Bedrock IDs often have a region prefix like "us." -- strip it.
    if head in {"us", "eu", "apac"}:
        head = model_id.split(".")[1].lower()
    if head.startswith("anthropic") or "claude" in model_id:
        return "anthropic"
    if head == "deepseek":
        return "deepseek_r1" if "r1" in model_id else "deepseek_v3"
    if head == "meta" or "llama" in model_id:
        return "llama"
    if head in PROVIDERS:
        return head
    raise ValueError(f"Cannot infer provider for model_id={model_id!r}; pass provider= explicitly.")


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #

class BedrockClient:
    """Thin, retry-aware Bedrock invoke_model wrapper.

    Examples
    --------
    >>> client = BedrockClient(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0")
    >>> client.generate(system_prompt="You are helpful.", user_prompt="Hi.")
    """

    def __init__(
        self,
        model_id: str,
        *,
        provider: str | None = None,
        region_name: str = "us-west-2",
        max_tokens: int = 1024,
        temperature: float = 0.0,
        top_p: float | None = None,
        thinking_budget: int | None = None,
        max_retries: int = 3,
        retry_backoff: float = 20.0,
    ) -> None:
        # Lazy import so the package is usable without boto3 installed.
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "BedrockClient requires boto3. Install with `pip install symbolic_drift[bedrock]`."
            ) from exc

        self.model_id = model_id
        self.provider = provider or infer_provider(model_id)
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider {self.provider!r}; choose from {sorted(PROVIDERS)}.")
        self.spec = PROVIDERS[self.provider]
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_budget = thinking_budget
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff
        self._client = boto3.client("bedrock-runtime", region_name=region_name)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a single completion. Returns the response text (never a dict)."""
        body_kwargs: dict[str, Any] = {
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            **self.spec.body_kwargs,
        }
        if self.spec.needs_thinking_kwarg:
            body_kwargs["thinking_budget"] = self.thinking_budget

        body = json.dumps(self.spec.build_body(system_prompt, user_prompt, **body_kwargs))

        response_body = self._invoke_with_retry(body)
        if self.spec.needs_thinking_kwarg:
            return self.spec.parse_response(response_body, thinking=self.thinking_budget is not None)
        return self.spec.parse_response(response_body)

    def _invoke_with_retry(self, body: str) -> dict[str, Any]:
        # Lazy import — only needed inside the retry loop.
        from botocore.exceptions import ClientError

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = self._client.invoke_model(body=body, modelId=self.model_id)
                return json.loads(response["body"].read())
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                # Only retry on transient errors.
                if code not in {"ThrottlingException", "ServiceUnavailableException",
                                "ModelTimeoutException", "InternalServerException"}:
                    raise
                last_exc = exc
            except Exception as exc:  # pragma: no cover  (network / json issues)
                last_exc = exc

            sleep_for = self.retry_backoff * (attempt + 1)
            logger.warning(
                "Bedrock invoke failed (attempt %d/%d): %s. Sleeping %.0fs.",
                attempt + 1, self.max_retries, last_exc, sleep_for,
            )
            time.sleep(sleep_for)

        assert last_exc is not None
        raise last_exc


# --------------------------------------------------------------------------- #
# Convenience factories used by the evaluation scripts.
# --------------------------------------------------------------------------- #

MODEL_ALIASES: dict[str, dict[str, str]] = {
    "claude_sonnet_4_5": {"provider": "anthropic", "model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"},
    "claude_haiku_4_5": {"provider": "anthropic", "model_id": "us.anthropic.claude-haiku-4-5-20251001-v1:0"},
    "gpt_oss_120b":     {"provider": "openai",    "model_id": "openai.gpt-oss-120b-1:0"},
    "deepseek_r1":      {"provider": "deepseek_r1", "model_id": "us.deepseek.r1-v1:0"},
    "deepseek_v3":      {"provider": "deepseek_v3", "model_id": "deepseek.v3-v1:0"},
    "qwen3_235b":       {"provider": "qwen",      "model_id": "qwen.qwen3-235b-a22b-2507-v1:0"},
    "llama3_70b":       {"provider": "llama",     "model_id": "meta.llama3-70b-instruct-v1:0"},
    "mistral_large_3":  {"provider": "mistral",   "model_id": "mistral.mistral-large-3-675b-instruct"},
    "moonshot_k2":      {"provider": "moonshot",  "model_id": "moonshot.kimi-k2-thinking"},
    "minimax_m2":       {"provider": "minimax",   "model_id": "us.minimax.minimax-m2"},
}


def build_client(model_alias: str | None = None, *, model_id: str | None = None,
                 provider: str | None = None, **kwargs: Any) -> BedrockClient:
    """Build a ``BedrockClient`` from either an alias or an explicit model_id."""
    if model_id is None:
        if model_alias is None or model_alias not in MODEL_ALIASES:
            raise ValueError(
                f"Pass model_id= explicitly or use one of: {sorted(MODEL_ALIASES)}."
            )
        spec = MODEL_ALIASES[model_alias]
        model_id, provider = spec["model_id"], spec["provider"]
    return BedrockClient(model_id=model_id, provider=provider, **kwargs)
