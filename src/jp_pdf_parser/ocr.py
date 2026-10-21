"""PaddleOCR-VL 1.6 region recognition; application IDs stay outside the model."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from .config import OCRSettings


@dataclass
class OCRPrediction:
    text: str
    raw: dict


class OCRBackend(Protocol):
    def recognize(self, path: Path, task: Literal["ocr", "table"]) -> OCRPrediction: ...


def decode_paddle_result(payload: dict) -> str:
    """Reject unexpected result formats rather than silently returning an empty string."""
    blocks = payload["res"]["parsing_res_list"]
    if not isinstance(blocks, list):
        raise ValueError("PaddleOCR parsing_res_list must be a list")
    texts = [block["block_content"] for block in blocks]
    if any(not isinstance(text, str) for text in texts):
        raise ValueError("PaddleOCR block_content must be a string")
    return "\n".join(texts)


class PaddleOCR:
    def __init__(self, settings: OCRSettings):
        self.settings = settings
        self._pipeline = None
        self._load_error: str | None = None

    def _load(self):
        if self._load_error:
            raise RuntimeError(self._load_error)
        if self._pipeline is None:
            try:
                # This pipeline loads a pinned local snapshot, not a model registry alias.
                os.environ.setdefault(
                    "PADDLE_PDX_CACHE_HOME",
                    str(
                        Path(self.settings.cache_dir).resolve().parent / "paddlex",
                    ),
                )
                os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
                from huggingface_hub import snapshot_download
                from paddleocr import PaddleOCRVL

                directory = snapshot_download(
                    self.settings.model_id,
                    revision=self.settings.revision,
                    cache_dir=self.settings.cache_dir,
                )
                self._pipeline = PaddleOCRVL(
                    pipeline_version="v1.6",
                    device=self.settings.device,
                    vl_rec_model_dir=directory,
                    use_layout_detection=False,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_queues=False,
                )
            except Exception as exc:
                self._load_error = f"PaddleOCR-VL initialization failed: {exc}"
                raise RuntimeError(self._load_error) from exc
        return self._pipeline

    def recognize(self, path: Path, task: Literal["ocr", "table"]) -> OCRPrediction:
        results = self._load().predict(
            input=str(path),
            use_layout_detection=False,
            prompt_label=task,
            max_new_tokens=self.settings.max_new_tokens,
            max_pixels=self.settings.max_pixels,
        )
        if len(results) != 1:
            raise ValueError("Expected exactly one PaddleOCR result for a crop")
        payload = results[0].json
        return OCRPrediction(text=decode_paddle_result(payload), raw=payload)
