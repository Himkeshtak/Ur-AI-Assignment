import json
from pathlib import Path

import pytest

from experiments.no_tatr import (
    CleanReading,
    ExperimentDocument,
    ExperimentPage,
    extract_tables,
    validate_experiment,
)
from jp_pdf_parser.config import Config, GroqSettings
from jp_pdf_parser.japanese import GroqReviewer

HTML = (
    "<table><tr><td>品目</td><td>数量</td><td>単価</td><td>金額</td></tr>"
    "<tr><td>紙</td><td>2</td><td>1,000</td><td>2,000</td></tr>"
    "<tr><td colspan='3'>合計</td><td>2,000</td></tr></table>"
)


def page_with_text(text=HTML):
    page = ExperimentPage(
        number=1,
        width_points=500,
        height_points=300,
        rotation=0,
        width_pixels=500,
        height_pixels=300,
        dpi=72,
        image_to_pdf=(1, 0, 0, 0, 1, 0),
        image_path="page.png",
    )
    reading = CleanReading(request_id="p1-clean-a1", image_path="page.png", text=text)
    page.annotated_reading = reading
    extract_tables(page, reading, 500)
    return page


def test_direct_table_has_spans_and_text_without_invented_pixel_geometry():
    page = page_with_text()
    table = page.tables[0]
    assert len(table.rows) == 3 and len(table.columns) == 4 and len(table.cells) == 10
    assert table.cells[-2].column_span == 3
    assert table.bbox is None and table.table_to_page is None and not table.geometry_valid
    assert all(c.bbox is None and c.table_bbox is None for c in table.cells)
    assert all(a.bbox is None and a.score is None for a in table.rows + table.columns)
    document = ExperimentDocument(source_name="test.pdf", pages=[page])
    serialized = document.model_dump_json()
    assert "tatr_grid_reconstruction" not in serialized
    assert ExperimentDocument.model_validate_json(serialized).all_cells()[-1].raw_text == "2,000"


@pytest.mark.parametrize(
    "text",
    [
        "not a table",
        "<table><tr><td>unclosed",
        "<table><tr><td>A</td><td>B</td></tr><tr><td>C</td></tr></table>",
    ],
)
def test_invalid_or_incomplete_topology_is_not_accepted(text):
    with pytest.raises(ValueError):
        page_with_text(text)


def test_existing_financial_rules_work_without_tatr_boxes():
    config = Config.load(Path("config.toml"))
    config.financial.mode = "invoice"
    document = ExperimentDocument(source_name="invoice.pdf", pages=[page_with_text()])
    validate_experiment(document, config)
    assert all(c.status == "passed" for c in document.checks if c.rule in {"line_amount", "total"})
    assert any(c.rule == "total" and c.observed == "2000" for c in document.checks)
    assert document.all_cells()[-1].raw_text == "2,000"
    assert document.all_cells()[-1].number == "2000"
    assert document.issues[0].code == "no_independent_geometry"


def test_existing_japanese_reviewer_uses_one_request_with_experiment_cells(tmp_path):
    reviewer = GroqReviewer(GroqSettings(), tmp_path / ".env")
    reviewer._load = lambda: None
    reviewer._fits = lambda messages: True
    requests = []

    def generate(messages):
        requests.append(messages)
        cells = json.loads(messages[1]["content"])["cells"]
        return json.dumps({"reviewed_cell_ids": [c["id"] for c in cells], "findings": []})

    reviewer._generate = generate
    document = ExperimentDocument(source_name="test.pdf", pages=[page_with_text()])
    review = reviewer.review(document)
    assert len(requests) == 1 and review.report.completed
    assert review.report.reviewed_cell_ids == [c.id for c in document.all_cells()]
