import json
import re

from conftest import ROOT, structure_objects, write_pdf
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
        return structure_objects()


class FixtureOCR:
    def __init__(self, table, fail_cell=None):
        self.values = json.loads(
            (ROOT / "tests/fixtures/simple_invoice.json").read_text(encoding="utf-8")
        )["cells"]
        self.html = html_for(table)
        self.fail_cell = fail_cell
        self.calls = []

    def recognize(self, path, task):
        self.calls.append((path.name, task))
        if task == "table":
            text = self.html
        else:
            if self.fail_cell and self.fail_cell in path.stem:
                raise RuntimeError("Deliberate OCR failure for one test cell")
            row, column = map(int, re.search(r"-r(\d+)-c(\d+)$", path.stem).groups())
            text = self.values[row - 1][column - 1]
        return OCRPrediction(
            text=text, raw={"res": {"parsing_res_list": [{"block_content": text}]}}
        )


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
    assert sum(task == "ocr" for _, task in ocr.calls) == 12
    serialized = json.loads((result.directory / "result.json").read_text(encoding="utf-8"))
    assert serialized["pages"][0]["tables"][0]["cells"][7]["raw_text"] == "2,000"
    manifest = json.loads((result.directory / "manifest.json").read_text())
    assert manifest["source_sha256"] == sha256(pdf)
    assert manifest["backends"]["ocr"].endswith("FixtureOCR")
    assert "pipeline.py" in manifest["application_source_sha256"]
    assert manifest["artifact_sha256"]["result.json"] == sha256(result.directory / "result.json")
    stages = [item["stage"] for item in manifest["timings"]]
    assert stages.index("K.validate") < stages.index("M.japanese") < stages.index("L.export")
    assert (result.directory / "tables/p1-t1/numbered.png").is_file()
    assert (result.directory / "schema.json").is_file()


def test_failed_cell_does_not_shift_other_responses(config, pdf, tmp_path, table):
    ocr = FixtureOCR(table, fail_cell="-r2-c2")
    result = Pipeline(
        config, structure=FixtureStructure(), ocr=ocr, japanese=FixtureReviewer()
    ).run(pdf, tmp_path / "output")
    assert result.document.status == "needs_review"
    cells = {(c.row, c.column): c for c in result.document.all_cells()}
    assert cells[2, 2].raw_text is None
    assert len(cells[2, 2].readings) == 2
    assert cells[2, 3].raw_text == "1,000"
    assert cells[2, 4].raw_text == "2,000"


def test_rejected_ocr_preserves_evidence_without_accepting_cell_value(config, pdf, tmp_path, table):
    class NoisyOCR(FixtureOCR):
        def recognize(self, path, task):
            if task == "ocr" and "-r2-c2" in path.stem:
                return OCRPrediction(
                    text="noise" * 1000,
                    raw={"original": "noise" * 1000},
                    error="Cell OCR exceeded character limit",
                )
            return super().recognize(path, task)

    result = Pipeline(
        config, structure=FixtureStructure(), ocr=NoisyOCR(table), japanese=FixtureReviewer()
    ).run(pdf, tmp_path / "output")
    cell = next(c for c in result.document.all_cells() if c.row == 2 and c.column == 2)
    assert cell.raw_text is None
    assert len(cell.readings) == 2
    assert all(reading.error for reading in cell.readings)
    assert (result.directory / cell.readings[0].raw_path).is_file()
    assert result.document.status == "needs_review"


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
