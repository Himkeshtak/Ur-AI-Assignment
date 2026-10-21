"""Small bounded chat transport. Credentials and HTTP headers never enter artifacts."""

import os
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from .config import APISettings


class APIError(RuntimeError):
    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        self.code = code


def credential(name: str, env_file: Path) -> str:
    value = os.environ.get(name)
    if not value:
        value = dotenv_values(env_file, encoding="utf-8-sig", interpolate=False).get(name)
    if not value or not value.strip():
        raise APIError(f"Missing {name}; set it in the environment or {env_file}")
    return value.strip()


class ChatAPI:
    def __init__(
        self,
        endpoint: str,
        api_key_env: str,
        env_file: Path,
        settings: APISettings,
        *,
        transport: httpx.BaseTransport | None = None,
    ):
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.env_file = env_file
        self.settings = settings
        self.transport = transport
        self.requests = 0
        self._last_request = 0.0

    def check_credentials(self) -> None:
        credential(self.api_key_env, self.env_file)

    def complete(self, payload: dict) -> tuple[str, dict]:
        key = credential(self.api_key_env, self.env_file)
        if self.requests >= self.settings.max_requests:
            raise APIError("Configured API request budget exhausted; no request was sent")
        delay = self.settings.min_interval_seconds - (time.monotonic() - self._last_request)
        if delay > 0:
            time.sleep(delay)
        self.requests += 1
        self._last_request = time.monotonic()
        try:
            # No automatic retries: a timed-out request might already have been billed.
            with httpx.Client(
                timeout=self.settings.timeout_seconds,
                follow_redirects=False,
                transport=self.transport,
            ) as client:
                response = client.post(
                    self.endpoint,
                    headers={"Authorization": f"Bearer {key}"},
                    json=payload,
                )
        except httpx.RequestError:
            raise APIError(
                "API connection failed or timed out; no automatic retry was sent"
            ) from None
        if response.status_code != 200:
            # Provider bodies may echo credentials or document text; never put them in errors.
            code = None
            try:
                candidate = response.json().get("error", {}).get("code")
                if candidate in ("json_validate_failed", "context_length_exceeded"):
                    code = candidate
            except (ValueError, AttributeError, TypeError):
                pass
            detail = f" ({code})" if code else ""
            raise APIError(
                f"API returned HTTP {response.status_code}{detail}; no fallback model was used",
                code=code,
            )
        try:
            raw = response.json()
            if raw.get("error"):
                raise ValueError("Provider error envelope")
            if raw["model"].casefold() != payload["model"].casefold():
                raise APIError("API returned an unexpected model; response was rejected")
            choices = raw["choices"]
            if len(choices) != 1:
                raise ValueError("Expected one completion")
            choice = choices[0]
            if choice.get("finish_reason") != "stop":
                raise APIError("API completion was truncated or did not finish normally")
            text = choice["message"]["content"]
            if not isinstance(text, str):
                raise ValueError("Expected text content")
        except (ValueError, KeyError, TypeError, AttributeError):
            raise APIError("API returned an invalid chat completion") from None
        return text, raw
