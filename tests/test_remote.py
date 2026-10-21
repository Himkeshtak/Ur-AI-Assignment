import base64
import io
import json

import httpx
from conftest import ROOT
from PIL import Image
from test_api import completion

from jp_pdf_parser.config import Config, GroqSettings, VLMRunSettings
from jp_pdf_parser.japanese import GroqReviewer, ReviewRun
from jp_pdf_parser.models import Document, JapaneseReview, Page
from jp_pdf_parser.ocr import VLMRunOCR
from jp_pdf_parser.pipeline import Pipeline


def test_shipped_config_selects_remote_models_and_cpu_tatr():
    config = Config.load(ROOT / "config.toml")
    pipeline = Pipeline(config)
    assert isinstance(pipeline.ocr, VLMRunOCR)
    assert isinstance(pipeline.japanese, GroqReviewer)
    assert config.detection.device == config.structure.device == "cpu"
    assert "revision" not in config.japanese.model_dump()


def test_vlmrun_resizes_the_complete_image_without_cropping_and_preserves_otsl(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("VLMRUN_API_KEY", "test-vlm-key")
    path = tmp_path / "cell.png"
    Image.new("RGB", (200, 100), "white").save(path)
    settings = VLMRunSettings(max_pixels=1024, min_interval_seconds=0)
    adapter = VLMRunOCR(settings, tmp_path / ".env")
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert set(payload) == {"model", "method", "messages"}
        uri = payload["messages"][0]["content"][0]["image_url"]["url"]
        with Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))) as image:
            assert image.width * image.height <= 1024
        return httpx.Response(200, json=completion(settings.model_id, "<fcel>2,000<nl>"))

    adapter.api.transport = httpx.MockTransport(handler)
    result = adapter.recognize(path, "table")
    assert result.text.startswith("<table>") and "2,000" in result.text
    assert result.raw["response"]["choices"][0]["message"]["content"] == "<fcel>2,000<nl>"
    assert result.raw["original_size"] == [200, 100]
    assert result.raw["sent_to_original"][0] > 1
    assert "test-vlm-key" not in json.dumps(result.raw)
    assert requests[0]["method"] == "table"


def test_unparseable_annotated_table_is_preserved_for_review(tmp_path, monkeypatch):
    monkeypatch.setenv("VLMRUN_API_KEY", "test-vlm-key")
    path = tmp_path / "table.png"
    Image.new("RGB", (50, 50)).save(path)
    adapter = VLMRunOCR(VLMRunSettings(), tmp_path / ".env")
    adapter.api.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json=completion(adapter.settings.model_id, "uncertain table text")
        )
    )
    result = adapter.recognize(path, "table")
    assert result.text == "uncertain table text"
    assert "table_conversion_error" in result.raw


def test_groq_strict_json_checks_complete_ids_and_records_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    adapter = GroqReviewer(GroqSettings(min_interval_seconds=0), tmp_path / ".env")

    def handler(request):
        payload = json.loads(request.content)
        assert payload["reasoning_effort"] == "none"
        schema = payload["response_format"]["json_schema"]
        assert schema["strict"] is True
        assert "suggestion" in schema["schema"]["$defs"]["Finding"]["required"]
        return httpx.Response(
            200,
            json=completion(adapter.settings.model_id, '{"reviewed_cell_ids":["a"],"findings":[]}'),
        )

    adapter.api.transport = httpx.MockTransport(handler)
    run = ReviewRun(report=JapaneseReview(completed=True))
    adapter._review_request([{"id": "a", "text": "用紙 2,000"}], {}, run)
    assert run.report.completed and run.report.reviewed_cell_ids == ["a"]
    assert run.traces[0]["generation"]["response"]["model"] == adapter.settings.model_id
    assert "test-groq-key" not in json.dumps(run.traces)


def test_groq_does_not_retry_payment_failure_or_change_model(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    adapter = GroqReviewer(GroqSettings(min_interval_seconds=0), tmp_path / ".env")
    adapter.api.transport = httpx.MockTransport(lambda request: httpx.Response(402))
    run = ReviewRun(report=JapaneseReview(completed=True))
    adapter._review_request([{"id": "a", "text": "用紙"}], {}, run)
    assert not run.report.completed and adapter.api.requests == 1
    assert "HTTP 402" in run.report.issues[0].message


def test_groq_repairs_provider_json_rejection_once(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    adapter = GroqReviewer(
        GroqSettings(min_interval_seconds=0, format_retries=1, max_requests=2), tmp_path / ".env"
    )
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if len(calls) == 1:
            return httpx.Response(400, json={"error": {"code": "json_validate_failed"}})
        return httpx.Response(
            200,
            json=completion(adapter.settings.model_id, '{"reviewed_cell_ids":["a"],"findings":[]}'),
        )

    adapter.api.transport = httpx.MockTransport(handler)
    run = ReviewRun(report=JapaneseReview(completed=True))
    adapter._review_request([{"id": "a", "text": "!"}], {}, run)
    assert run.report.completed and run.report.reviewed_cell_ids == ["a"]
    assert len(calls) == 2 and len(calls[1]["messages"]) > len(calls[0]["messages"])


def test_oversized_ocr_is_retained_as_evidence_but_not_accepted(tmp_path, monkeypatch):
    monkeypatch.setenv("VLMRUN_API_KEY", "test-vlm-key")
    path = tmp_path / "cell.png"
    Image.new("RGB", (50, 50)).save(path)
    adapter = VLMRunOCR(VLMRunSettings(max_page_characters=32), tmp_path / ".env")
    adapter.api.transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200, json=completion(adapter.settings.model_id, "hallucination" * 100)
        )
    )
    result = adapter.recognize(path, "ocr")
    assert result.error and result.text == "hallucination" * 100
    assert result.raw["response"]["choices"][0]["message"]["content"] == result.text


def test_oversized_cell_is_flagged_without_truncation_or_inference(table, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-groq-key")
    adapter = GroqReviewer(
        GroqSettings(context_tokens=4096, min_interval_seconds=0), tmp_path / ".env"
    )
    table.cells = table.cells[:1]
    table.cells[0].raw_text = "用紙" * 5000
    document = Document(
        source_name="synthetic",
        pages=[
            Page(
                number=1,
                width_points=424,
                height_points=184,
                rotation=0,
                width_pixels=424,
                height_pixels=184,
                dpi=72,
                image_to_pdf=(1, 0, 0, 0, 1, 0),
                image_path="synthetic.png",
                tables=[table],
            )
        ],
    )
    run = adapter.review(document)
    assert not run.report.completed and adapter.api.requests == 0
    assert run.report.issues[0].code == "review_context_limit"
    assert table.cells[0].raw_text == "用紙" * 5000


def test_whole_document_review_uses_one_compact_request(table, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    adapter = GroqReviewer(GroqSettings(min_interval_seconds=0), tmp_path / ".env")
    tables = []
    for i in range(20):
        copy = table.model_copy(deep=True)
        copy.id = f"p1-t{i + 1}"
        for cell in copy.cells:
            cell.id = cell.id.replace("p1-t1-", f"{copy.id}-")
        tables.append(copy)
    page = Page(
        number=1,
        width_points=500,
        height_points=300,
        rotation=0,
        width_pixels=500,
        height_pixels=300,
        dpi=72,
        image_to_pdf=(1, 0, 0, 0, 1, 0),
        image_path="page.png",
        tables=tables,
    )
    document = Document(source_name="many-cells.pdf", pages=[page])
    requests = []

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        cells = json.loads(payload["messages"][1]["content"])["cells"]
        assert len(cells) == 240
        assert cells[0]["id"] == "c1" and cells[-1]["id"] == "c240"
        return httpx.Response(
            200,
            json=completion(
                adapter.settings.model_id,
                json.dumps(
                    {
                        "reviewed_cell_ids": [c["id"] for c in cells],
                        "findings": [
                            {
                                "cell_id": "c1",
                                "code": "uncertain_label",
                                "message": "check",
                                "suggestion": None,
                            }
                        ],
                    }
                ),
            ),
        )

    adapter.api.transport = httpx.MockTransport(handler)
    run = adapter.review(document)
    assert len(requests) == 1 and run.report.completed
    assert run.report.reviewed_cell_ids == [c.id for c in document.all_cells()]
    assert run.report.issues[0].cell_ids == [document.all_cells()[0].id]
    assert run.traces[0]["review_id_map"]["c1"] == document.all_cells()[0].id


def test_default_single_request_review_does_not_retry_json_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    adapter = GroqReviewer(GroqSettings(min_interval_seconds=0), tmp_path / ".env")
    adapter.api.transport = httpx.MockTransport(
        lambda _: httpx.Response(400, json={"error": {"code": "json_validate_failed"}})
    )
    run = ReviewRun(report=JapaneseReview(completed=True))
    adapter._review_request([{"id": "a", "text": "x"}], {}, run)
    assert adapter.api.requests == 1
    assert not run.report.completed


def test_unmapped_page_does_not_send_empty_cells_to_groq(table, tmp_path, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    for cell in table.cells:
        cell.raw_text = None
    page = Page(
        number=1,
        width_points=500,
        height_points=300,
        rotation=0,
        width_pixels=500,
        height_pixels=300,
        dpi=72,
        image_to_pdf=(1, 0, 0, 0, 1, 0),
        image_path="page.png",
        tables=[table],
    )
    adapter = GroqReviewer(GroqSettings(), tmp_path / ".env")
    run = adapter.review(Document(source_name="unmapped.pdf", pages=[page]))
    assert adapter.api.requests == 0
    assert not run.report.completed
    assert run.report.issues[0].code == "no_reviewable_cells"
