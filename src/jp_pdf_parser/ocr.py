"""PaddleOCR-VL 1.6 full-page recognition; application IDs stay outside the model."""

import base64
import io
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from PIL import Image

from .api import ChatAPI
from .artifacts import sha256
from .config import OCRSettings, VLMRunSettings
from .table_format import otsl_to_html


@dataclass
class OCRPrediction:
    text: str
    raw: dict
    error: str | None = None


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
            raise ValueError("Expected exactly one PaddleOCR result for a page")
        payload = results[0].json
        return OCRPrediction(text=decode_paddle_result(payload), raw=payload)


class VLMRunOCR:
    """Explicit 1.6 image requests; no page detector, implicit LLM, or model fallback."""

    def __init__(self, settings: VLMRunSettings, env_file: Path):
        self.settings = settings
        self.api = ChatAPI(
            "https://gateway.vlm.run/v1/openai/chat/completions",
            settings.api_key_env,
            env_file,
            settings,
        )

    def recognize(self, path: Path, task: Literal["ocr", "table"]) -> OCRPrediction:
        if task not in ("ocr", "table"):
            raise ValueError("VLM Run task must be ocr or table")
        with Image.open(path) as original:
            image = original.convert("RGB")
        original_size = image.size
        if image.width * image.height > self.settings.max_pixels:
            scale = math.sqrt(self.settings.max_pixels / (image.width * image.height))
            resized = image.resize(
                (max(1, math.floor(image.width * scale)), max(1, math.floor(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
            image.close()
            image = resized
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        sent_size = image.size
        image.close()
        content = base64.b64encode(buffer.getvalue()).decode("ascii")
        text, response = self.api.complete(
            {
                "model": self.settings.model_id,
                "method": task,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{content}"},
                            }
                        ],
                    }
                ],
                # Live catalog advertises no generation parameters for this model.
                # Do not send unsupported temperature/max_tokens or a postprocessing LLM.
            }
        )
        raw = {
            "provider": "vlmrun",
            "method": task,
            "response": response,
            "input_sha256": sha256(path),
            "original_size": list(original_size),
            "sent_size": list(sent_size),
            "sent_to_original": [
                original_size[0] / sent_size[0],
                0,
                0,
                0,
                original_size[1] / sent_size[1],
                0,
            ],
        }
        if task == "table":
            try:
                converted = otsl_to_html(text)
                raw["table_conversion"] = "otsl_to_html"
                text = converted
            except ValueError as exc:
                # Preserve the original reading; validation will flag unknown topology.
                raw["table_conversion_error"] = str(exc)
        error = None
        if len(text) > self.settings.max_page_characters:
            error = (
                "Page OCR exceeded the configured character limit; raw output retained for review"
            )
        return OCRPrediction(text=text, raw=raw, error=error)
