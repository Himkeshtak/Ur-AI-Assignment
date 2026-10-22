"""The approved A-L pipeline, with the Japanese validator M between K and L."""

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .artifacts import Artifacts
from .config import Config, GroqSettings, VLMRunSettings
from .grid import annotated_page, area, intersection_area, reconstruct_grid, suppress_duplicates
from .japanese import GroqReviewer, JapaneseBackend, JapaneseValidator
from .models import Document, Issue, JapaneseReview, Page, Reading
from .ocr import OCRBackend, PaddleOCR, VLMRunOCR
from .pdf import inspect_source, render_source
from .structure import TATR, StructureBackend
from .tables import attach_page_reading
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
        api_counts = {
            name: getattr(getattr(backend, "api", None), "requests", None)
            for name, backend in {"ocr": self.ocr, "japanese": self.japanese}.items()
        }
        try:
            with artifacts.stage("A.inspect"):
                document.source_sha256, document.page_count = inspect_source(
                    source, self.config.render
                )
            pages = render_source(source, self.config.render, artifacts)
            for page in pages:
                document.pages.append(page)
                try:
                    with Image.open(artifacts.path(page.image_path)) as image:
                        self._page(image, page, artifacts)
                except Exception as exc:
                    page.issues.append(
                        Issue(code="page_failed", stage="C-J", severity="error", message=str(exc))
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
            artifacts.write(
                "request-counts.json",
                {
                    name: (
                        backend.api.requests - api_counts[name]
                        if api_counts[name] is not None
                        else None
                    )
                    for name, backend in {"ocr": self.ocr, "japanese": self.japanese}.items()
                },
            )
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

    def _page(self, image: Image.Image, page: Page, artifacts: Artifacts) -> None:
        prefix = f"pages/p{page.number:03d}"
        with artifacts.stage(f"C.detect.p{page.number}"):
            detections = self.structure.detect(image)
        artifacts.write(
            f"{prefix}-detections.json", [d.model_dump(mode="json") for d in detections]
        )
        detections = suppress_duplicates(
            [
                d
                for d in detections
                if d.score >= self.config.detection.threshold
                and d.label in {"table", "table rotated"}
            ]
        )
        detections.sort(key=lambda d: (d.bbox[1], d.bbox[0]))
        if not detections:
            return
        # Both TATR passes receive the complete original page. No image is cropped.
        try:
            with artifacts.stage(f"D.structure.p{page.number}"):
                objects = self.structure.recognize(image)
        except Exception as exc:
            page.issues.append(Issue(code="structure_failed", stage="D", message=str(exc)))
            objects = []
        artifacts.write(f"{prefix}-structure.json", [o.model_dump(mode="json") for o in objects])
        for index, detection in enumerate(detections, 1):
            x0, y0, x1, y1 = detection.bbox
            region = (max(0, x0), max(0, y0), min(image.width, x1), min(image.height, y1))
            if region[0] >= region[2] or region[1] >= region[3]:
                page.issues.append(
                    Issue(code="invalid_table_box", stage="D", message="Table outside page")
                )
                continue
            owned = []
            ambiguous = False
            for obj in objects:
                owners = [
                    d
                    for d in detections
                    if intersection_area(obj.bbox, d.bbox) / area(obj.bbox) > 0.5
                ]
                if detection in owners:
                    if len(owners) == 1:
                        owned.append(obj)
                    elif obj.score >= self.config.structure.threshold and obj.label != "table":
                        ambiguous = True
            table = reconstruct_grid(
                f"p{page.number}-t{index}",
                page.number,
                page.image_path,
                image.size,
                (1, 0, 0, 0, 1, 0),
                owned,
                self.config.structure.threshold,
                self.config.render.max_cells_per_table,
                region=region,
            )
            if ambiguous or detection.label == "table rotated":
                table.geometry_valid = False
                table.issues.append(
                    Issue(
                        code="unsupported_page_structure",
                        stage="E",
                        severity="error",
                        message=(
                            "Ambiguous table ownership or rotation requires layout review; "
                            "no crop fallback was used"
                        ),
                    )
                )
            page.tables.append(table)
            artifacts.write(f"tables/{table.id}/grid-g1.json", table)
        if not page.tables:
            return
        relative = f"{prefix}-annotated.png"
        with artifacts.stage(f"F.overlay.p{page.number}"):
            with annotated_page(image, page.tables, self.config.render.max_pixels) as overlay:
                overlay.save(artifacts.path(relative))
        reading = Reading(
            request_id=f"p{page.number}-annotated-a1",
            variant="annotated_page",
            image_path=relative,
            source_bbox_in_image=(0, 0, image.width, image.height),
            image_to_page=(1, 0, 0, 0, 1, 0),
        )
        page.annotated_reading = reading
        try:
            with artifacts.stage(f"H.{reading.request_id}"):
                prediction = self.ocr.recognize(artifacts.path(relative), "table")
            reading.raw_path = f"ocr/{reading.request_id}.json"
            artifacts.write(reading.raw_path, prediction.raw)
            reading.text, reading.error = prediction.text, prediction.error
        except Exception as exc:
            reading.error = str(exc)
        with artifacts.stage(f"J.assemble.p{page.number}"):
            attach_page_reading(page)
        artifacts.write(f"{prefix}-ocr-request.json", reading)
