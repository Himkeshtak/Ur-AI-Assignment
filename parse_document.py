"""Run TATR, PaddleOCR-VL and Japanese review on a PDF or image."""

import argparse
import json
from pathlib import Path

from pydantic import Field

from jp_pdf_parser.artifacts import sha256, write_json
from jp_pdf_parser.config import Config, GroqSettings, VLMRunSettings
from jp_pdf_parser.japanese import JapaneseBackend
from jp_pdf_parser.models import Axis, Box, Document, Issue, Record, Status
from jp_pdf_parser.ocr import OCRBackend
from jp_pdf_parser.pipeline import Pipeline
from jp_pdf_parser.structure import StructureBackend

ROOT = Path(__file__).resolve().parent


class ResultCell(Record):
    id: str
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    text: str | None
    number: str | None = None
    bbox: Box


class ResultTable(Record):
    id: str
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    bbox: Box
    geometry_valid: bool
    rows: list[Axis]
    columns: list[Axis]
    cells: list[ResultCell]


class ResultPage(Record):
    page: int = Field(ge=1)
    image: str
    annotated_image: str | None
    width: int = Field(ge=1)
    height: int = Field(ge=1)
    tables: list[ResultTable]


class Result(Record):
    schema_version: str = "2.0"
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
    for spec in (config.detection, config.structure):
        if spec.local_dir is not None:
            spec.local_dir = (path.parent / spec.local_dir).resolve()
        spec.cache_dir = str((path.parent / spec.cache_dir).resolve())
    if not isinstance(config.ocr, VLMRunSettings) or not isinstance(config.japanese, GroqSettings):
        raise ValueError("This script requires the VLM Run and Groq backends")
    # Enforce the public command's request policy without editing config.toml.
    config.japanese.format_retries = 0
    config.japanese.max_requests = 1
    return config


def simplify(document: Document) -> Result:
    return Result(
        source=document.source_name,
        status=document.status,
        japanese_review_completed=bool(
            document.japanese_review and document.japanese_review.completed
        ),
        issues=document.all_issues(),
        pages=[
            ResultPage(
                page=p.number,
                image=p.image_path,
                annotated_image=p.annotated_reading.image_path if p.annotated_reading else None,
                width=p.width_pixels,
                height=p.height_pixels,
                tables=[
                    ResultTable(
                        id=t.id,
                        row_count=len(t.rows),
                        column_count=len(t.columns),
                        bbox=t.bbox,
                        geometry_valid=t.geometry_valid,
                        rows=t.rows,
                        columns=t.columns,
                        cells=[
                            ResultCell(
                                id=c.id,
                                row=c.row,
                                column=c.column,
                                row_span=c.row_span,
                                column_span=c.column_span,
                                text=c.raw_text,
                                number=c.number,
                                bbox=c.bbox,
                            )
                            for c in t.cells
                        ],
                    )
                    for t in p.tables
                ],
            )
            for p in document.pages
        ],
    )


def run(
    source: Path,
    config: Config,
    output: Path,
    *,
    structure: StructureBackend | None = None,
    ocr: OCRBackend | None = None,
    japanese: JapaneseBackend | None = None,
) -> tuple[Path, Result]:
    """Use the production pipeline; only the additional compact export lives here."""
    completed = Pipeline(config, structure=structure, ocr=ocr, japanese=japanese).run(
        source, output
    )
    result = simplify(completed.document)
    directory = completed.directory
    write_json(directory / "simple.json", result)
    write_json(directory / "simple.schema.json", Result.model_json_schema())
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name in ("simple.json", "simple.schema.json"):
        manifest["artifact_sha256"][name] = sha256(directory / name)
    manifest["entrypoint"] = {
        "script": "parse_document.py",
        "sha256": sha256(Path(__file__)),
        "route": "tatr-full-architecture",
    }
    write_json(manifest_path, manifest)
    return directory, result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="PDF or single image")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "documents")
    parser.add_argument("--config", type=Path, default=ROOT / "config.toml")
    parser.add_argument("--financial-mode", choices=("table", "invoice"), default="table")
    args = parser.parse_args()
    try:
        print("Running TATR -> PaddleOCR-VL -> validation -> Japanese review...", flush=True)
        directory, result = run(
            args.input, load_config(args.config, args.financial_mode), args.output_dir
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    print(
        f"Status: {result.status}\nJSON: {directory / 'result.json'}\n"
        f"Schema: {directory / 'schema.json'}\n"
        f"Simple JSON: {directory / 'simple.json'}\n"
        f"Simple schema: {directory / 'simple.schema.json'}"
    )
    for page in result.pages:
        print(f"PNG: {directory / page.image}")
        if page.annotated_image:
            print(f"Boxes PNG: {directory / page.annotated_image}")
    return {"checks_passed": 0, "needs_review": 2, "failed": 1}[result.status]


if __name__ == "__main__":
    raise SystemExit(main())
