import pytest

from jp_pdf_parser.japanese import JapaneseValidator, ReviewRun, validate_review_response
from jp_pdf_parser.models import JapaneseReview


def test_review_rejects_unknown_missing_and_duplicate_ids():
    for text in (
        '{"reviewed_cell_ids":[],"findings":[]}',
        '{"reviewed_cell_ids":["a","a"],"findings":[]}',
        '{"reviewed_cell_ids":["a"],"findings":[{"cell_id":"b","code":"suspicious_text","message":"x"}]}',
        '```json\n{"reviewed_cell_ids":["a"],"findings":[]}\n```',
    ):
        with pytest.raises(ValueError):
            validate_review_response(text, ["a"])


def test_valid_review_uses_cell_references():
    response = validate_review_response('{"reviewed_cell_ids":["a"],"findings":[]}', ["a"])
    assert not response.findings


def test_invalid_model_json_is_retried_once_then_incomplete(config):
    validator = JapaneseValidator(config.japanese)
    attempts = []
    validator._fits = lambda messages: True
    validator._generate = lambda messages: attempts.append(messages) or "invalid JSON"
    run = ReviewRun(report=JapaneseReview(completed=True))
    validator._review_request([{"id": "a", "text": "用紙"}], {}, run)
    assert len(attempts) == 2
    assert not run.report.completed
    assert run.report.issues[0].code == "invalid_japanese_review"


def test_suggestions_are_findings_and_not_applied(config):
    validator = JapaneseValidator(config.japanese)
    validator._generate = lambda messages: (
        '{"reviewed_cell_ids":["a"],"findings":['
        '{"cell_id":"a","code":"suspicious_text","message":"確認","suggestion":"用紙"}]}'
    )
    run = ReviewRun(report=JapaneseReview(completed=True))
    cells = [{"id": "a", "text": "用紐"}]
    validator._review_request(cells, {}, run)
    assert cells[0]["text"] == "用紐"
    assert "用紙" in run.report.issues[0].message


def test_generation_failure_preserves_previous_chunk_evidence(config):
    validator = JapaneseValidator(config.japanese)

    def fail(messages):
        raise RuntimeError("Model generation failed")

    validator._generate = fail
    run = ReviewRun(report=JapaneseReview(completed=True, reviewed_cell_ids=["previous"]))
    validator._review_request([{"id": "a", "text": "用紙"}], {}, run)
    assert not run.report.completed
    assert run.report.reviewed_cell_ids == ["previous"]
    assert run.traces[0]["error"] == "Model generation failed"
