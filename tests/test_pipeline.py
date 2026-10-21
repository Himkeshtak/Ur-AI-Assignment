import json

from conftest import structure_objects, write_pdf
from PIL import Image
from test_tables import html_for

from jp_pdf_parser.artifacts import sha256
from jp_pdf_parser.japanese import ReviewRun
from jp_pdf_parser.models import Detection, JapaneseReview
from jp_pdf_parser.ocr import OCRPrediction
from jp_pdf_parser.pipeline import Pipeline


class FixtureStructure:
    def detect(self, image):
        return [Detection(label="table", score=0.99, bbox=(50, 50, 450, 210))]

    def recognize(self, image):
        return [
            obj.model_copy(update={"bbox": tuple(v + 38 for v in obj.bbox)})
            for obj in structure_objects()
        ]


class FixtureOCR:
    def __init__(self, table):
        self.html = html_for(table)
        self.calls = []

    def recognize(self, path, task):
        self.calls.append((path.name, task))
        assert task == "table"
        return OCRPrediction(text=self.html, raw={"table_html": self.html})


class FixtureReviewer:
    def __init__(self):
        self.saw_financial_checks = False

    def review(self, document):
        self.saw_financial_checks = bool(document.checks)
        return ReviewRun(
            report=JapaneseReview(
                completed=True, reviewed_cell_ids=[c.id for c in document.all_cells()]
            )
        )


def test_pipeline_renders_and_exports_with_injected_models(config, pdf, tmp_path, table):
    reviewer, ocr = FixtureReviewer(), FixtureOCR(table)
    result = Pipeline(config, structure=FixtureStructure(), ocr=ocr, japanese=reviewer).run(
        pdf, tmp_path / "output"
    )
    assert result.document.status == "checks_passed"
    assert reviewer.saw_financial_checks
    assert len(result.document.all_cells()) == 12
    assert sum(task == "table" for _, task in ocr.calls) == 1
    assert len(ocr.calls) == 1
    assert not list(result.directory.rglob("cells"))
    assert not list(result.directory.rglob("source.png"))
    serialized = json.loads((result.directory / "result.json").read_text(encoding="utf-8"))
    assert serialized["pages"][0]["tables"][0]["cells"][7]["raw_text"] == "2,000"
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["source_sha256"] == sha256(pdf)
    assert manifest["backends"]["ocr"].endswith("FixtureOCR")
    assert "pipeline.py" in manifest["application_source_sha256"]
    assert manifest["artifact_sha256"]["result.json"] == sha256(result.directory / "result.json")
    stages = [item["stage"] for item in manifest["timings"]]
    assert stages.index("K.validate") < stages.index("M.japanese") < stages.index("L.export")
    with Image.open(result.directory / "pages/p001.png") as original:
        with Image.open(result.directory / "pages/p001-annotated.png") as annotated:
            assert annotated.size == original.size == (500, 300)
    assert all(c.bbox == c.table_bbox for c in result.document.all_cells())
    assert (result.directory / "schema.json").is_file()


def test_failed_page_ocr_preserves_grid_without_cell_fallback(config, pdf, tmp_path, table):
    class FailedOCR(FixtureOCR):
        def recognize(self, path, task):
            self.calls.append((path.name, task))
            raise RuntimeError("Deliberate page OCR failure")

    ocr = FailedOCR(table)
    result = Pipeline(
        config, structure=FixtureStructure(), ocr=ocr, japanese=FixtureReviewer()
    ).run(pdf, tmp_path)
    assert result.document.status == "needs_review"
    assert len(ocr.calls) == 1
    assert all(c.raw_text is None for c in result.document.all_cells())
    assert result.document.pages[0].annotated_reading.error
    assert result.document.pages[0].issues[0].code == "page_alignment_uncertain"


def test_rejected_page_reading_retains_raw_evidence(config, pdf, tmp_path, table):
    class NoisyOCR(FixtureOCR):
        def recognize(self, path, task):
            return OCRPrediction(text=self.html, raw={"original": self.html}, error="Rejected OCR")

    result = Pipeline(
        config, structure=FixtureStructure(), ocr=NoisyOCR(table), japanese=FixtureReviewer()
    ).run(pdf, tmp_path)
    reading = result.document.pages[0].annotated_reading
    assert reading.error and reading.text
    assert (result.directory / reading.raw_path).is_file()
    assert all(c.raw_text is None for c in result.document.all_cells())


def test_two_pages_use_two_ocr_calls_and_one_review(config, tmp_path, table, monkeypatch):
    class CountingReviewer(FixtureReviewer):
        count = 0

        def review(self, document):
            self.count += 1
            return super().review(document)

    def forbid_crop(*args, **kwargs):
        raise AssertionError("Cropping is forbidden in the page pipeline")

    monkeypatch.setattr(Image.Image, "crop", forbid_crop)
    ocr, reviewer = FixtureOCR(table), CountingReviewer()
    result = Pipeline(config, structure=FixtureStructure(), ocr=ocr, japanese=reviewer).run(
        write_pdf(tmp_path / "two.pdf", pages=2), tmp_path / "output"
    )
    assert result.document.status == "checks_passed"
    assert len(ocr.calls) == 2 and reviewer.count == 1


def test_missing_pdf_still_exports_failure(config, tmp_path):
    result = Pipeline(config, japanese=FixtureReviewer()).run(tmp_path / "missing.pdf", tmp_path)
    assert result.document.status == "failed"
    assert (result.directory / "result.json").is_file()
    assert (result.directory / "manifest.json").is_file()


def test_partial_page_failure_cannot_pass(config, tmp_path, table):
    class PartialFailure(FixtureStructure):
        count = 0

        def detect(self, image):
            self.count += 1
            if self.count == 2:
                raise RuntimeError("Deliberate second-page failure")
            return super().detect(image)

    pdf = write_pdf(tmp_path / "two-pages.pdf", pages=2)
    result = Pipeline(
        config, structure=PartialFailure(), ocr=FixtureOCR(table), japanese=FixtureReviewer()
    ).run(pdf, tmp_path / "output")
    assert len(result.document.pages) == 2
    assert result.document.status == "needs_review"
    assert result.document.pages[1].issues[0].code == "page_failed"


def test_incomplete_japanese_review_cannot_pass(config, pdf, tmp_path, table):
    class IncompleteReviewer:
        def review(self, document):
            return ReviewRun(report=JapaneseReview(completed=True, reviewed_cell_ids=[]))

    result = Pipeline(
        config, structure=FixtureStructure(), ocr=FixtureOCR(table), japanese=IncompleteReviewer()
    ).run(pdf, tmp_path)
    assert result.document.status == "needs_review"
    assert not result.document.japanese_review.completed


def test_runs_never_overwrite_one_another(config, tmp_path):
    pipeline = Pipeline(config, japanese=FixtureReviewer())
    a = pipeline.run(tmp_path / "missing.pdf", tmp_path)
    b = pipeline.run(tmp_path / "missing.pdf", tmp_path)
    assert a.directory != b.directory
    assert (a.directory / "result.json").is_file()


def test_valid_pdf_with_no_detected_tables_requires_review(config, pdf, tmp_path):
    class NoTables(FixtureStructure):
        def detect(self, image):
            return []

    result = Pipeline(config, structure=NoTables(), japanese=FixtureReviewer()).run(pdf, tmp_path)
    assert result.document.status == "needs_review"
    assert result.document.pages[0].issues[0].code == "no_tables"


def test_mismatched_full_page_stays_reviewable_and_retains_raw_text(config, pdf, tmp_path, table):
    ocr = FixtureOCR(table)
    ocr.html = "<table><tr><td>unassigned text</td></tr></table>"
    result = Pipeline(
        config, structure=FixtureStructure(), ocr=ocr, japanese=FixtureReviewer()
    ).run(pdf, tmp_path)
    assert result.document.status == "needs_review"
    assert all(c.raw_text is None for c in result.document.all_cells())
    assert "unassigned text" in result.document.pages[0].annotated_reading.text
    assert len(ocr.calls) == 1


def test_multiple_tables_still_send_one_page_request(config, pdf, tmp_path, table):
    class TwoTables(FixtureStructure):
        def detect(self, image):
            return [
                Detection(label="table", score=1, bbox=(10, 10, 210, 120)),
                Detection(label="table", score=1, bbox=(260, 150, 460, 280)),
            ]

    ocr = FixtureOCR(table)
    result = Pipeline(config, structure=TwoTables(), ocr=ocr, japanese=FixtureReviewer()).run(
        pdf, tmp_path
    )
    assert len(result.document.pages[0].tables) == 2
    assert len(ocr.calls) == 1
    assert result.document.status == "needs_review"
