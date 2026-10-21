import json
from pathlib import Path

import httpx
import pytest

from jp_pdf_parser.api import APIError, ChatAPI, credential
from jp_pdf_parser.config import APISettings


def completion(model="test-model", text="2,000", finish="stop"):
    return {
        "id": "request-1",
        "model": model,
        "choices": [
            {"finish_reason": finish, "message": {"content": text}},
        ],
        "usage": {"total_tokens": 10},
    }


def client(monkeypatch, handler, **settings):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "test-secret")
    return ChatAPI(
        "https://provider.example/chat/completions",
        "TEST_PROVIDER_KEY",
        Path("not-used.env"),
        APISettings(min_interval_seconds=0, **settings),
        transport=httpx.MockTransport(handler),
    )


def test_credentials_are_literal_and_environment_takes_precedence(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text('TEST_LITERAL_KEY="${UNEXPANDED}-value"\n', encoding="utf-8")
    monkeypatch.delenv("TEST_LITERAL_KEY", raising=False)
    assert credential("TEST_LITERAL_KEY", path) == "${UNEXPANDED}-value"
    monkeypatch.setenv("TEST_LITERAL_KEY", "environment-value")
    assert credential("TEST_LITERAL_KEY", path) == "environment-value"
    with pytest.raises(APIError, match="Missing MISSING_TEST_KEY"):
        credential("MISSING_TEST_KEY", path)


def test_transport_preserves_raw_result_without_request_headers(monkeypatch):
    def handler(request):
        assert request.headers["authorization"] == "Bearer test-secret"
        return httpx.Response(200, json=completion())

    api = client(monkeypatch, handler, max_requests=1)
    text, raw = api.complete({"model": "test-model"})
    assert text == "2,000"
    assert "test-secret" not in json.dumps(raw)
    assert raw["usage"]["total_tokens"] == 10
    with pytest.raises(APIError, match="budget exhausted"):
        api.complete({"model": "test-model"})
    assert api.requests == 1


@pytest.mark.parametrize("status", [301, 400, 401, 402, 403, 429, 500])
def test_http_errors_are_sanitized_and_never_retry_or_fallback(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="test-secret reflected by upstream")

    api = client(monkeypatch, handler)
    with pytest.raises(APIError) as error:
        api.complete({"model": "test-model"})
    assert "test-secret" not in str(error.value)
    assert str(status) in str(error.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "payload",
    [
        completion(finish="length"),
        completion(model="different-model"),
        completion(text=None),
        {"choices": []},
        {"error": {"message": "test-secret"}},
    ],
)
def test_incomplete_or_wrong_model_responses_are_rejected(monkeypatch, payload):
    api = client(monkeypatch, lambda request: httpx.Response(200, json=payload))
    with pytest.raises(APIError) as error:
        api.complete({"model": "test-model"})
    assert "test-secret" not in str(error.value)


def test_network_error_is_sanitized(monkeypatch):
    def handler(request):
        raise httpx.ReadTimeout("test-secret", request=request)

    api = client(monkeypatch, handler)
    with pytest.raises(APIError, match="timed out") as error:
        api.complete({"model": "test-model"})
    assert "test-secret" not in str(error.value)


def test_request_spacing_is_enforced(monkeypatch):
    sleeps = []
    monkeypatch.setattr("jp_pdf_parser.api.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("jp_pdf_parser.api.time.sleep", sleeps.append)
    api = client(monkeypatch, lambda request: httpx.Response(200, json=completion()))
    api.settings.min_interval_seconds = 2.1
    api.complete({"model": "test-model"})
    api.complete({"model": "test-model"})
    assert sleeps == [2.1]
