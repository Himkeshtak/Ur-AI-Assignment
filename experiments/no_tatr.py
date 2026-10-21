"""Isolated clean-page PaddleOCR experiment. Production modules are read-only."""

import argparse
import csv
import json
import unicodedata
from pathlib import Path
from typing import Literal

from pydantic import Field

from jp_pdf_parser.artifacts import Artifacts, sha256
from jp_pdf_parser.config import Config, GroqSettings, VLMRunSettings
from jp_pdf_parser.japanese import GroqReviewer
from jp_pdf_parser.models import (
    Affine,
    Axis,
    Box,
    Cell,
    Check,
    Document,
    Issue,
    JapaneseReview,
    Page,
    Reading,
    Table,
)
from jp_pdf_parser.ocr import VLMRunOCR
from jp_pdf_parser.pdf import inspect_pdf, render_pages
from jp_pdf_parser.tables import _TableHTML
from jp_pdf_parser.validation import final_status, validate_document, validate_invoice


class CleanReading(Reading):
    variant: Literal["clean_page"] = "clean_page"


class LogicalAxis(Axis):
    bbox: Box | None = None
    coordinate_frame: Literal["unavailable"] = "unavailable"
    score: float | None = None


class LogicalCell(Cell):
    bbox: Box | None = None
    table_bbox: Box | None = None
    coordinate_frame: Literal["unavailable"] = "unavailable"
    bbox_source: Literal["unavailable"] = "unavailable"
    readings: list[CleanReading] = Field(default_factory=list)


class LogicalTable(Table):
    bbox: Box | None = None
    coordinate_frame: Literal["unavailable"] = "unavailable"
    structure_coordinate_frame: Literal["otsl_positions"] = "otsl_positions"
    table_to_page: Affine | None = None
    rows: list[LogicalAxis] = Field(default_factory=list)
    columns: list[LogicalAxis] = Field(default_factory=list)
    cells: list[LogicalCell] = Field(default_factory=list)


class ExperimentPage(Page):
    annotated_reading: CleanReading | None = None
    tables: list[LogicalTable] = Field(default_factory=list)


class ExperimentDocument(Document):
    schema_version: Literal["experiment.no-tatr.1"] = "experiment.no-tatr.1"
    pages: list[ExperimentPage] = Field(default_factory=list)


def extract_tables(page: ExperimentPage, reading: CleanReading, limit: int) -> None:
    """Use Paddle's explicit row/column/span topology, without inventing geometry."""
    if reading.error or not reading.text:
        raise ValueError(reading.error or "No OCR text returned")
    parser = _TableHTML()
    parser.feed(reading.text)
    parser.close()
    if parser.cells is not None or not parser.tables:
        raise ValueError("OCR did not return complete table HTML")
    tables = []
    for index, parsed in enumerate(parser.tables, 1):
        if not parsed or len(parsed) > limit:
            raise ValueError("OCR cell count is outside the experiment limit")
        rows = max(r + rs - 1 for r, _, rs, _ in parsed)
        columns = max(c + cs - 1 for _, c, _, cs in parsed)
        if rows * columns > limit:
            raise ValueError("OCR grid exceeds the experiment limit")
        covered = {
            (y, x) for r, c, rs, cs in parsed for y in range(r, r + rs) for x in range(c, c + cs)
        }
        if covered != {(r, c) for r in range(1, rows + 1) for c in range(1, columns + 1)}:
            raise ValueError("OCR table has missing grid positions")
        prefix = f"p{page.number}-t{index}-ocr"
        table = LogicalTable(
            id=f"p{page.number}-t{index}",
            page_number=page.number,
            image_path=page.image_path,
            rows=[LogicalAxis(id=f"{prefix}-r{r}", index=r) for r in range(1, rows + 1)],
            columns=[LogicalAxis(id=f"{prefix}-c{c}", index=c) for c in range(1, columns + 1)],
        )
        for (row, column, rs, cs), text in sorted(parsed.items()):
            table.cells.append(
                LogicalCell(
                    id=f"{prefix}-r{row}-c{column}",
                    row=row,
                    column=column,
                    row_span=rs,
                    column_span=cs,
                    row_ids=[f"{prefix}-r{r}" for r in range(row, row + rs)],
                    column_ids=[f"{prefix}-c{c}" for c in range(column, column + cs)],
                    raw_text=text,
                    normalized_text=unicodedata.normalize("NFKC", text).strip(),
                    readings=[reading.model_copy(update={"text": text})],
                )
            )
        tables.append(table)
    page.tables = tables


def validate_experiment(document: ExperimentDocument, config: Config) -> None:
    # The shared validator skips physical geometry because it is unavailable.
    validate_document(document, config.financial)
    document.issues.append(
        Issue(
            code="no_independent_geometry",
            stage="K",
            message="Paddle supplies logical topology only; pixel alignment is not verified",
        )
    )
    document.checks.append(
        Check(rule="physical_geometry", status="not_applicable", detail="No TATR or cell boxes")
    )
    for page in document.pages:
        for table in page.tables:
            document.checks.append(
                Check(
                    rule="ocr_topology",
                    status="passed",
                    cell_ids=[c.id for c in table.cells],
                    detail="Explicit OCR spans cover a rectangular logical grid; not a pixel check",
                )
            )
            # Reuse the exact financial rules, including generic-table exclusions.
            document.checks.extend(validate_invoice(table, config.financial))
            for cell in table.cells:
                text = cell.raw_text or ""
                if not text.strip():
                    table.issues.append(
                        Issue(
                            code="empty_or_unreadable",
                            stage="K",
                            cell_ids=[cell.id],
                            message="Empty OCR does not establish a blank cell",
                        )
                    )
                elif "\ufffd" in text or any(
                    unicodedata.category(char) == "Cc" and char not in "\n\r\t" for char in text
                ):
                    table.issues.append(
                        Issue(
                            code="suspicious_encoding",
                            stage="K",
                            cell_ids=[cell.id],
                            message="Text contains replacement/control characters",
                        )
                    )


def production_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "config.toml",
        *sorted((root / "src").rglob("*.py")),
        *sorted((root / "scripts").rglob("*.py")),
    ]
    return {str(p.relative_to(root)): sha256(p) for p in paths}


def table_stats(table: Table) -> dict:
    return {
        "id": table.id,
        "rows": len(table.rows),
        "columns": len(table.columns),
        "cells": len(table.cells),
        "text_cells": sum(c.raw_text is not None for c in table.cells),
        "geometry_valid": table.geometry_valid,
        "topology": sorted((c.row, c.column, c.row_span, c.column_span) for c in table.cells),
    }


def compare(document: ExperimentDocument, baseline: Path | None, config: Config) -> dict:
    result = {
        "experiment": [table_stats(t) for p in document.pages for t in p.tables],
        "accuracy_proven": False,
        "caveat": (
            "This removes TATR, annotations and fixed-grid rejection together. "
            "One document, without ground truth, cannot isolate a sole cause."
        ),
    }
    if baseline is None:
        return result
    original = Document.model_validate_json((baseline / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((baseline / "manifest.json").read_text(encoding="utf-8"))
    if original.source_sha256 != document.source_sha256:
        raise ValueError("Baseline PDF hash differs; comparison rejected")
    if manifest["config"]["render"] != config.render.model_dump(mode="json"):
        raise ValueError("Baseline render settings differ; comparison rejected")
    for key in ("model_id", "max_pixels"):
        if manifest["config"]["ocr"][key] != getattr(config.ocr, key):
            raise ValueError(f"Baseline OCR {key} differs; comparison rejected")
    result["baseline_directory"] = str(baseline)
    result["baseline_tatr"] = [table_stats(t) for p in original.pages for t in p.tables]
    clean = {(p.number, t.id): t for p in document.pages for t in p.tables}
    result["clean_response_against_saved_tatr_grid"] = [
        {
            "table_id": t.id,
            "tatr_geometry_valid": t.geometry_valid,
            "exact_topology_match": (
                table_stats(t)["topology"] == table_stats(clean[p.number, t.id])["topology"]
                if (p.number, t.id) in clean
                else False
            ),
        }
        for p in original.pages
        for t in p.tables
    ]
    return result


def run(source: Path, config: Config, output: Path, baseline: Path | None = None) -> Path:
    if not isinstance(config.ocr, VLMRunSettings) or not isinstance(config.japanese, GroqSettings):
        raise ValueError("Experiment requires the configured VLM Run and Groq backends")
    if config.japanese.format_retries != 0 or config.japanese.max_requests != 1:
        raise ValueError("Experiment requires one Groq attempt, without automatic retries")
    before = production_hashes()
    artifacts = Artifacts(output)
    document = ExperimentDocument(source_name=source.name)
    ocr, reviewer = (
        VLMRunOCR(config.ocr, config.env_file),
        GroqReviewer(config.japanese, config.env_file),
    )
    document.source_sha256, document.page_count = inspect_pdf(source, config.render)
    if baseline is not None:
        compare(
            document, baseline, config
        )  # Reject incompatible controls before spending API calls.
    for rendered in render_pages(source, config.render, artifacts):
        page = ExperimentPage.model_validate(rendered.model_dump())
        document.pages.append(page)
        reading = CleanReading(
            request_id=f"p{page.number}-clean-a1",
            image_path=page.image_path,
            source_bbox_in_image=(0, 0, page.width_pixels, page.height_pixels),
            image_to_page=(1, 0, 0, 0, 1, 0),
        )
        page.annotated_reading = reading
        try:
            with artifacts.stage(f"H.clean-page.p{page.number}"):
                prediction = ocr.recognize(artifacts.path(page.image_path), "table")
            reading.text, reading.error = prediction.text, prediction.error
            reading.raw_path = f"ocr/{reading.request_id}.json"
            artifacts.write(reading.raw_path, prediction.raw)
            extract_tables(page, reading, config.render.max_cells_per_table)
        except Exception as exc:
            page.issues.append(Issue(code="direct_ocr_failed", stage="H-J", message=str(exc)))
        artifacts.write(f"pages/p{page.number:03d}-ocr-request.json", reading)
    with artifacts.stage("K.validate"):
        validate_experiment(document, config)
    try:
        with artifacts.stage("M.japanese"):
            review = reviewer.review(document)
        document.japanese_review = review.report
        expected = {cell.id for cell in document.all_cells()}
        actual = review.report.reviewed_cell_ids
        if len(actual) != len(set(actual)) or set(actual) != expected:
            document.japanese_review.completed = False
            document.japanese_review.issues.append(
                Issue(code="review_coverage", stage="M", message="Incomplete cell review coverage")
            )
        artifacts.write("japanese-traces.json", review.traces)
    except Exception as exc:
        document.japanese_review = JapaneseReview(
            issues=[Issue(code="japanese_review_failed", stage="M", message=str(exc))]
        )
    document.status = final_status(document)
    with artifacts.stage("L.export"):
        artifacts.write("result.json", document)
        artifacts.write("schema.json", ExperimentDocument.model_json_schema())
        artifacts.write("japanese-review.json", document.japanese_review)
        artifacts.write(
            "validation.json",
            {
                "status": document.status,
                "checks": [c.model_dump(mode="json") for c in document.checks],
                "issues": [i.model_dump(mode="json") for i in document.all_issues()],
            },
        )
        artifacts.write("comparison.json", compare(document, baseline, config))
        artifacts.write(
            "request-counts.json", {"ocr": ocr.api.requests, "japanese": reviewer.api.requests}
        )
        with artifacts.path("cells.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(
                ["table_id", "cell_id", "row", "column", "row_span", "column_span", "raw_text"]
            )
            for page in document.pages:
                for table in page.tables:
                    for cell in table.cells:
                        writer.writerow(
                            [
                                table.id,
                                cell.id,
                                cell.row,
                                cell.column,
                                cell.row_span,
                                cell.column_span,
                                cell.raw_text,
                            ]
                        )
        unchanged = before == production_hashes()
        artifacts.write(
            "experiment.json",
            {
                "name": "direct-clean-page-no-tatr",
                "geometry_available": False,
                "script_sha256": sha256(Path(__file__)),
                "production_unchanged": unchanged,
                "production_sha256": before,
                "baseline": str(baseline) if baseline else None,
            },
        )
        if not unchanged:
            raise RuntimeError("Production files changed during the experiment")
    artifacts.finish(
        config,
        document.source_sha256,
        {
            "structure": "none; Paddle OTSL topology only",
            "ocr": "jp_pdf_parser.ocr.VLMRunOCR",
            "japanese": "jp_pdf_parser.japanese.GroqReviewer",
        },
    )
    return artifacts.root


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output/experiments/no-tatr"))
    parser.add_argument("--financial-mode", choices=("invoice", "table"), default="table")
    args = parser.parse_args()
    config = Config.load(args.config)
    config.financial.mode = args.financial_mode
    directory = run(args.pdf, config, args.output_dir, args.baseline)
    print(directory)


if __name__ == "__main__":
    main()
