"""The approved A-L pipeline, with the Japanese validator M between K and L."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .artifacts import Artifacts
from .config import Config, GroqSettings, VLMRunSettings
from .grid import (
    OVERLAY_MARGIN,
    cell_crop,
    cell_pixel_box,
    numbered_overlay,
    reconstruct_grid,
    suppress_duplicates,
    translate_input,
)
from .japanese import GroqReviewer, JapaneseBackend, JapaneseValidator
from .models import Document, Issue, JapaneseReview, Reading, Table
from .ocr import OCRBackend, PaddleOCR, VLMRunOCR
from .pdf import crop_table, inspect_pdf, render_pages
from .structure import TATR, StructureBackend
from .tables import attach_readings
from .validation import final_status, validate_document


@dataclass
class RunResult:
    document: Document
    directory: Path


class Pipeline:
    def __init__(
        self,
        config: Config,
        *,
        structure: StructureBackend | None = None,
        ocr: OCRBackend | None = None,
        japanese: JapaneseBackend | None = None,
    ):
        self.config = config
        self.structure = structure or TATR(config.detection, config.structure)
        self.ocr = ocr or (
            VLMRunOCR(config.ocr, config.env_file)
            if isinstance(config.ocr, VLMRunSettings)
            else PaddleOCR(config.ocr)
        )
        self.japanese = japanese or (
            GroqReviewer(config.japanese, config.env_file)
            if isinstance(config.japanese, GroqSettings)
            else JapaneseValidator(config.japanese)
        )

    def run(self, source: Path, output_root: Path) -> RunResult:
        artifacts = Artifacts(output_root)
        document = Document(source_name=source.name)
        try:
            with artifacts.stage("A.inspect"):
                document.source_sha256, document.page_count = inspect_pdf(
                    source, self.config.render
                )
            pages = render_pages(source, self.config.render, artifacts)
            for page in pages:
                document.pages.append(page)
                try:
                    with Image.open(artifacts.path(page.image_path)) as image:
                        with artifacts.stage(f"C.detect.p{page.number}"):
                            detections = self.structure.detect(image)
                        artifacts.write(
                            f"pages/p{page.number:03d}-detections.json",
                            [d.model_dump(mode="json") for d in detections],
                        )
                        tables = suppress_duplicates(
                            [
                                d
                                for d in detections
                                if d.score >= self.config.detection.threshold
                                and d.label in {"table", "table rotated"}
                            ]
                        )
                        tables.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
                        for index, detection in enumerate(tables, 1):
                            table_id = f"p{page.number}-t{index}"
                            try:
                                crop, transform = crop_table(
                                    image,
                                    detection,
                                    self.config.render.table_padding,
                                )
                                with crop:
                                    table = self._table(
                                        table_id, page.number, crop, transform, artifacts
                                    )
                                page.tables.append(table)
                            except Exception as exc:
                                page.issues.append(
                                    Issue(
                                        code="table_failed",
                                        stage="D-I",
                                        severity="error",
                                        message=f"{table_id}: {exc}",
                                    )
                                )
                except Exception as exc:
                    page.issues.append(
                        Issue(code="page_failed", stage="C", severity="error", message=str(exc))
                    )
        except Exception as exc:
            document.issues.append(
                Issue(code="processing_failed", stage="A-B", severity="error", message=str(exc))
            )
        try:
            with artifacts.stage("K.validate"):
                validate_document(document, self.config.financial)
        except Exception as exc:
            document.issues.append(
                Issue(code="validation_failed", stage="K", severity="error", message=str(exc))
            )
        # M is intentionally after K, and before any final document serialization.
        try:
            with artifacts.stage("M.japanese"):
                review = self.japanese.review(document)
            document.japanese_review = review.report
            expected = {cell.id for cell in document.all_cells()}
            actual = review.report.reviewed_cell_ids
            if len(actual) != len(set(actual)) or set(actual) != expected:
                document.japanese_review.completed = False
                document.japanese_review.issues.append(
                    Issue(
                        code="review_coverage",
                        stage="M",
                        message="Japanese review must cover every cell exactly once",
                    )
                )
            artifacts.write("japanese-review.json", review.report)
            artifacts.write("japanese-traces.json", review.traces)
        except Exception as exc:
            document.japanese_review = JapaneseReview(
                issues=[
                    Issue(
                        code="japanese_review_failed",
                        stage="M",
                        severity="error",
                        message=str(exc),
                    )
                ]
            )
            artifacts.write("japanese-review.json", document.japanese_review)
        document.status = final_status(document)
        with artifacts.stage("L.export"):
            artifacts.write("schema.json", Document.model_json_schema())
            artifacts.write(
                "validation.json",
                {
                    "status": document.status,
                    "issues": [issue.model_dump(mode="json") for issue in document.all_issues()],
                    "checks": [check.model_dump(mode="json") for check in document.checks],
                },
            )
            artifacts.write("result.json", document)
        artifacts.finish(
            self.config,
            document.source_sha256,
            {
                name: f"{type(backend).__module__}.{type(backend).__qualname__}"
                for name, backend in {
                    "structure": self.structure,
                    "ocr": self.ocr,
                    "japanese": self.japanese,
                }.items()
            },
        )
        return RunResult(document=document, directory=artifacts.root)

    def _read(self, request_id: str, relative: str, variant: str, artifacts: Artifacts) -> Reading:
        reading = Reading(request_id=request_id, variant=variant, image_path=relative)
        try:
            task = "table" if variant == "annotated_table" else "ocr"
            with artifacts.stage(f"{'H' if task == 'table' else 'I'}.{request_id}"):
                prediction = self.ocr.recognize(artifacts.path(relative), task)
            raw_path = f"ocr/{request_id}.json"
            artifacts.write(raw_path, prediction.raw)
            reading.raw_path, reading.text = raw_path, prediction.text
            reading.error = prediction.error
        except Exception as exc:
            reading.error = str(exc)
        return reading

    def _table(self, table_id, page_number, crop, transform, artifacts: Artifacts) -> Table:
        relative = f"tables/{table_id}/source.png"
        crop.save(artifacts.path(relative))
        with artifacts.stage(f"D.structure.{table_id}"):
            objects = self.structure.recognize(crop)
        artifacts.write(
            f"tables/{table_id}/structure.json", [obj.model_dump(mode="json") for obj in objects]
        )
        with artifacts.stage(f"E.grid.{table_id}"):
            table = reconstruct_grid(
                table_id,
                page_number,
                relative,
                crop.size,
                transform,
                objects,
                self.config.structure.threshold,
                self.config.render.max_cells_per_table,
            )
        artifacts.write(f"tables/{table_id}/grid-g1.json", table)
        if not table.geometry_valid:
            return table
        try:
            with artifacts.stage(f"F.overlay.{table_id}"):
                overlay = numbered_overlay(crop, table, self.config.render.max_pixels)
                with overlay:
                    overlay_path = f"tables/{table_id}/numbered.png"
                    overlay.save(artifacts.path(overlay_path))
            table.annotated_reading = self._read(
                f"{table_id}-annotated-a1",
                overlay_path,
                "annotated_table",
                artifacts,
            )
            table.annotated_reading.source_bbox_in_image = (
                OVERLAY_MARGIN,
                OVERLAY_MARGIN,
                OVERLAY_MARGIN + crop.width,
                OVERLAY_MARGIN + crop.height,
            )
            table.annotated_reading.image_to_page = translate_input(
                transform,
                -OVERLAY_MARGIN,
                -OVERLAY_MARGIN,
            )
        except Exception as exc:
            table.issues.append(Issue(code="overlay_failed", stage="F", message=str(exc)))
        retry_budget = self.config.ocr.max_retries_per_table
        for cell in table.cells:
            relative = f"tables/{table_id}/cells/{cell.id}.png"
            try:
                with artifacts.stage(f"G.crop.{cell.id}"):
                    pixel_box = cell_pixel_box(crop, cell)
                    with cell_crop(crop, cell) as image:
                        image.save(artifacts.path(relative))
            except ValueError as exc:
                table.issues.append(
                    Issue(code="invalid_cell_crop", stage="G", cell_ids=[cell.id], message=str(exc))
                )
                continue
            reading = self._read(f"{cell.id}-a1", relative, "clean_cell", artifacts)
            reading.source_bbox_in_image = (
                0,
                0,
                pixel_box[2] - pixel_box[0],
                pixel_box[3] - pixel_box[1],
            )
            reading.image_to_page = translate_input(transform, pixel_box[0], pixel_box[1])
            cell.readings.append(reading)
            if (reading.error or not reading.text or not reading.text.strip()) and retry_budget:
                retry_budget -= 1
                reading = self._read(f"{cell.id}-a2", relative, "clean_cell", artifacts)
                reading.source_bbox_in_image = cell.readings[0].source_bbox_in_image
                reading.image_to_page = cell.readings[0].image_to_page
                cell.readings.append(reading)
            if reading.error:
                table.issues.append(
                    Issue(
                        code="cell_ocr_failed", stage="I", cell_ids=[cell.id], message=reading.error
                    )
                )
        with artifacts.stage(f"J.assemble.{table_id}"):
            attach_readings(table)
        artifacts.write(
            f"tables/{table_id}/ocr-requests.json",
            {
                "annotated": (
                    table.annotated_reading.model_dump(mode="json")
                    if table.annotated_reading
                    else None
                ),
                "cells": [r.model_dump(mode="json") for c in table.cells for r in c.readings],
            },
        )
        return table
