"""Parse a PDF or image into simple table JSON and complete-page PNG files."""

import argparse
from collections.abc import Iterator
from pathlib import Path

from PIL import Image, ImageOps
from pydantic import Field

from experiments.no_tatr import (
    CleanReading,
    ExperimentDocument,
    ExperimentPage,
    extract_tables,
    validate_experiment,
)
from jp_pdf_parser.artifacts import Artifacts, sha256
from jp_pdf_parser.config import Config, GroqSettings, VLMRunSettings
from jp_pdf_parser.japanese import GroqReviewer, JapaneseBackend
from jp_pdf_parser.models import Issue, JapaneseReview, Record, Status
from jp_pdf_parser.ocr import OCRBackend, VLMRunOCR
from jp_pdf_parser.pdf import inspect_pdf, render_pages
from jp_pdf_parser.validation import final_status

ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class ResultCell(Record):
    id: str
    row: int
    column: int
    row_span: int = 1
    column_span: int = 1
    text: str | None
    number: str | None = None


class ResultTable(Record):
    id: str
    row_count: int
    column_count: int
    cells: list[ResultCell]


class ResultPage(Record):
    page: int
    image: str
    width: int
    height: int
    tables: list[ResultTable]


class Result(Record):
    schema_version: str = "1.0"
    source: str
    status: Status
    japanese_review_completed: bool
    pages: list[ResultPage]
    issues: list[Issue] = Field(default_factory=list)


def load_config(path: Path, financial_mode: str) -> Config:
    path = path.resolve()
    config = Config.load(path)
    config.env_file = (path.parent / config.env_file).resolve()
    config.financial.mode = financial_mode
    if not isinstance(config.ocr, VLMRunSettings) or not isinstance(config.japanese, GroqSettings):
        raise ValueError("This script requires the VLM Run and Groq backends")
    # Enforce the public command's request policy without editing config.toml.
    config.japanese.format_retries = 0
    config.japanese.max_requests = 1
    return config


def source_pages(source: Path, config: Config, artifacts: Artifacts) -> Iterator[ExperimentPage]:
    if source.suffix.lower() == ".pdf":
        for page in render_pages(source, config.render, artifacts):
            yield ExperimentPage.model_validate(page.model_dump())
        return
    with artifacts.stage("B.normalize-image"), Image.open(source) as original:
        if getattr(original, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are unsupported; supply a PDF or single image")
        if original.width * original.height > config.render.max_pixels:
            raise ValueError("Image exceeds the configured pixel limit")
        with ImageOps.exif_transpose(original) as oriented:
            with oriented.convert("RGBA") as rgba, Image.new("RGB", rgba.size, "white") as image:
                image.paste(rgba, mask=rgba.getchannel("A"))
                image.save(artifacts.path("pages/p001.png"))
                page = ExperimentPage(
                    number=1, width_points=image.width, height_points=image.height,
                    rotation=0, width_pixels=image.width, height_pixels=image.height,
                    dpi=72, image_to_pdf=(1, 0, 0, 0, 1, 0), image_path="pages/p001.png",
                )
    # Raster input has a nominal 72-DPI virtual page; no source PDF geometry exists.
    yield page


def inspect_source(source: Path, config: Config) -> tuple[str, int]:
    if not source.is_file():
        raise ValueError(f"Input does not exist: {source}")
    if source.suffix.lower() == ".pdf":
        return inspect_pdf(source, config.render)
    if source.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Use a PDF, PNG, JPEG, BMP, TIFF or WebP file")
    if source.stat().st_size > config.render.max_file_mb * 1024 * 1024:
        raise ValueError("Input exceeds the configured file-size limit")
    with Image.open(source) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are unsupported; supply a PDF or single image")
        if image.width * image.height > config.render.max_pixels:
            raise ValueError("Image exceeds the configured pixel limit")
    return sha256(source), 1


def simplify(document: ExperimentDocument) -> Result:
    return Result(
        source=document.source_name, status=document.status,
        japanese_review_completed=bool(document.japanese_review and document.japanese_review.completed),
        issues=document.all_issues(),
        pages=[ResultPage(
            page=p.number, image=p.image_path, width=p.width_pixels, height=p.height_pixels,
            tables=[ResultTable(
                id=t.id, row_count=len(t.rows), column_count=len(t.columns),
                cells=[ResultCell(
                    id=c.id, row=c.row, column=c.column, row_span=c.row_span,
                    column_span=c.column_span, text=c.raw_text, number=c.number,
                ) for c in t.cells],
            ) for t in p.tables],
        ) for p in document.pages],
    )


def run(
    source: Path, config: Config, output: Path, *,
    ocr: OCRBackend | None = None, japanese: JapaneseBackend | None = None,
) -> tuple[Path, Result]:
    source_hash, page_count = inspect_source(source, config)
    ocr = ocr or VLMRunOCR(config.ocr, config.env_file)
    japanese = japanese or GroqReviewer(config.japanese, config.env_file)
    artifacts = Artifacts(output)
    document = ExperimentDocument(source_name=source.name, source_sha256=source_hash, page_count=page_count)
    try:
        for page in source_pages(source, config, artifacts):
            document.pages.append(page)
            print(f"Reading page {page.number}/{page_count}...", flush=True)
            reading = CleanReading(
                request_id=f"p{page.number}-clean-a1", image_path=page.image_path,
                source_bbox_in_image=(0, 0, page.width_pixels, page.height_pixels),
                image_to_page=(1, 0, 0, 0, 1, 0),
            )
            page.annotated_reading = reading
            try:
                with artifacts.stage(f"H.ocr.p{page.number}"):
                    prediction = ocr.recognize(artifacts.path(page.image_path), "table")
                reading.text, reading.error = prediction.text, prediction.error
                reading.raw_path = f"evidence/ocr/p{page.number:03d}.json"
                artifacts.write(reading.raw_path, prediction.raw)
                extract_tables(page, reading, config.render.max_cells_per_table)
            except Exception as exc:
                reading.error = str(exc)
                page.issues.append(Issue(code="page_ocr_failed", stage="H-J", severity="error", message=str(exc)))
    except Exception as exc:
        document.issues.append(Issue(code="render_failed", stage="B", severity="error", message=str(exc)))
    with artifacts.stage("K.validate"):
        validate_experiment(document, config)
    print("Reviewing extracted text...", flush=True)
    try:
        with artifacts.stage("M.japanese"):
            review = japanese.review(document)
        document.japanese_review = review.report
        ids = review.report.reviewed_cell_ids
        if len(ids) != len(set(ids)) or set(ids) != {c.id for c in document.all_cells()}:
            document.japanese_review.completed = False
            document.japanese_review.issues.append(Issue(code="review_coverage", stage="M", message="Incomplete cell review coverage"))
        artifacts.write("evidence/japanese-traces.json", review.traces)
    except Exception as exc:
        document.japanese_review = JapaneseReview(issues=[Issue(code="japanese_review_failed", stage="M", message=str(exc))])
    document.status = final_status(document)
    result = simplify(document)
    with artifacts.stage("L.export"):
        artifacts.write("result.json", result)
        artifacts.write("schema.json", Result.model_json_schema())
        artifacts.write("evidence/document.json", document)
        artifacts.write("evidence/japanese-review.json", document.japanese_review)
        artifacts.write("evidence/validation.json", {
            "checks": [c.model_dump(mode="json") for c in document.checks],
            "issues": [i.model_dump(mode="json") for i in document.all_issues()],
        })
        artifacts.write("evidence/runner.json", {
            "script_sha256": sha256(Path(__file__)), "route": "direct-page-no-tatr",
            "source_type": "pdf" if source.suffix.lower() == ".pdf" else "image",
            "raster_coordinate_note": "Image inputs use a nominal 72-DPI virtual page; no source PDF coordinates exist",
            "requests": {name: getattr(getattr(backend, "api", None), "requests", None)
                         for name, backend in {"ocr": ocr, "japanese": japanese}.items()},
        })
    artifacts.finish(config, source_hash, {
        "structure": "none; Paddle logical topology",
        "ocr": f"{type(ocr).__module__}.{type(ocr).__qualname__}",
        "japanese": f"{type(japanese).__module__}.{type(japanese).__qualname__}",
    })
    return artifacts.root, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF or single image")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "documents")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--financial-mode", choices=("table", "invoice"), default="table")
    args = parser.parse_args()
    try:
        directory, result = run(args.input, load_config(args.config, args.financial_mode), args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(f"Status: {result.status}\nJSON: {directory / 'result.json'}\nSchema: {directory / 'schema.json'}")
    for page in result.pages:
        print(f"PNG: {directory / page.image}")
    return {"checks_passed": 0, "needs_review": 2, "failed": 1}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
