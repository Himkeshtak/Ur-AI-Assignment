from types import SimpleNamespace

import pytest

from jp_pdf_parser.ocr import PaddleOCR, decode_paddle_result


def test_paddle_crop_uses_region_task_and_preserves_raw_json(config, tmp_path):
    raw = {"res": {"parsing_res_list": [{"block_content": "2,000"}]}}
    calls = []
    adapter = PaddleOCR(config.ocr)
    adapter._pipeline = SimpleNamespace(
        predict=lambda **kwargs: calls.append(kwargs) or [SimpleNamespace(json=raw)]
    )
    result = adapter.recognize(tmp_path / "cell.png", "ocr")
    assert result.text == "2,000" and result.raw is raw
    assert calls[0]["use_layout_detection"] is False
    assert calls[0]["prompt_label"] == "ocr"


def test_unknown_paddle_schema_does_not_look_like_empty_text():
    with pytest.raises(KeyError):
        decode_paddle_result({"unexpected": []})
