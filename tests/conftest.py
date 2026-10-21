import json
from pathlib import Path

import pytest

from jp_pdf_parser.config import Config, JapaneseSettings, OCRSettings
from jp_pdf_parser.grid import reconstruct_grid
from jp_pdf_parser.models import Detection

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config():
    result = Config.load(ROOT / "config.toml")
    # Legacy local-adapter contracts stay testable while the shipped config uses APIs.
    result.ocr = OCRSettings(
        model_id="PaddlePaddle/PaddleOCR-VL-1.6",
        revision="c5630abae1d940eafe0697512a0325494b02ab42",
    )
    result.japanese = JapaneseSettings(
        format_retries=1,
        model_id="llm-jp/llm-jp-3-1.8b-instruct",
        revision="413b7929df092690b0f9449f3045baaf8deba0a5",
    )
    result.render.dpi = 72
    return result


def structure_objects():
    objects = [Detection(label="table", score=0.99, bbox=(12, 12, 412, 172))]
    objects += [
        Detection(label="table row", score=0.99, bbox=(12, 12 + i * 40, 412, 52 + i * 40))
        for i in range(4)
    ]
    objects += [
        Detection(label="table column", score=0.99, bbox=(12 + i * 100, 12, 112 + i * 100, 172))
        for i in range(4)
    ]
    objects += [Detection(label="table column header", score=0.99, bbox=(12, 12, 412, 52))]
    objects += [
        Detection(label="table spanning cell", score=0.99, bbox=(12, 12 + i * 40, 312, 52 + i * 40))
        for i in (2, 3)
    ]
    return objects


@pytest.fixture
def table():
    result = reconstruct_grid(
        "p1-t1", 1, "table.png", (424, 184), (1, 0, 38, 0, 1, 38), structure_objects(), 0.6, 500
    )
    values = json.loads((ROOT / "tests/fixtures/simple_invoice.json").read_text(encoding="utf-8"))
    for cell in result.cells:
        cell.raw_text = values["cells"][cell.row - 1][cell.column - 1]
    return result


def write_pdf(path: Path, *, pages: int = 1) -> Path:
    """Small valid vector PDF, generated locally without a PDF authoring dependency."""
    content = b"BT /F1 12 Tf 50 250 Td (SYNTHETIC INVOICE) Tj ET\n"
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>", b""]
    page_ids = []
    for _ in range(pages):
        page_id, stream_id = len(objects) + 1, len(objects) + 2
        page_ids.append(page_id)
        objects.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 500 300] "
                "/Resources << /Font << /F1 << /Type /Font /Subtype /Type1 "
                f"/BaseFont /Helvetica >> >> >> /Contents {stream_id} 0 R >>"
            ).encode()
        )
        objects.append(
            b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"endstream"
        )
    kids = " ".join(f"{page_id} 0 R" for page_id in page_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {pages} >>".encode()
    data, offsets = bytearray(b"%PDF-1.4\n"), [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data.extend(f"{number} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref = len(data)
    data.extend(f"xref\n0 {len(offsets)}\n0000000000 65535 f \n".encode())
    for offset in offsets[1:]:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(
        f"trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    path.write_bytes(data)
    return path


@pytest.fixture
def pdf(tmp_path):
    return write_pdf(tmp_path / "synthetic.pdf")
