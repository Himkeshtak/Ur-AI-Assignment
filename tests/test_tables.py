import pytest

from jp_pdf_parser.models import Page, Reading
from jp_pdf_parser.tables import attach_page_reading, map_invoice


def html_for(table):
    rows = []
    for row in table.rows:
        cells = [c for c in table.cells if c.row == row.index]
        rows.append(
            "<tr>"
            + "".join(
                f'<td rowspan="{c.row_span}" colspan="{c.column_span}">{c.raw_text}</td>'
                for c in cells
            )
            + "</tr>"
        )
    return "<table>" + "".join(rows) + "</table>"


def test_invoice_mapping_uses_header_and_ids(table):
    view = map_invoice(table)
    assert view.columns == {"quantity": 2, "unit_price": 3, "amount": 4}
    assert view.lines[0].amount_id == "p1-t1-g1-r2-c4"
    assert view.total_id == "p1-t1-g1-r4-c4"


def page_for(tables, text):
    return Page(
        number=1,
        width_points=500,
        height_points=300,
        rotation=0,
        width_pixels=500,
        height_pixels=300,
        dpi=72,
        image_to_pdf=(1, 0, 0, 0, 1, 0),
        image_path="page.png",
        tables=tables,
        annotated_reading=Reading(
            request_id="p1-annotated-a1",
            variant="annotated_page",
            image_path="annotated.png",
            raw_path="ocr/page.json",
            text=text,
            image_to_page=(1, 0, 0, 0, 1, 0),
        ),
    )


def test_page_table_binds_exact_topology_and_shared_evidence(table):
    text = html_for(table)
    for cell in table.cells:
        cell.raw_text = None
    page = page_for([table], text)
    attach_page_reading(page)
    assert not page.issues
    assert next(c for c in table.cells if (c.row, c.column) == (2, 4)).raw_text == "2,000"
    assert all(c.readings[0].raw_path == "ocr/page.json" for c in table.cells)
    assert all(c.readings[0].variant == "annotated_page" for c in table.cells)


@pytest.mark.parametrize(
    "text",
    [
        "<table><tr><td>2,000</td></tr></table>",
        "<table><tr><td>unclosed",
        "<table><table><tr><td>nested</td></tr></table></table>",
        "<table><tr><td rowspan='2'>past end</td></tr></table>",
        "not a table",
    ],
)
def test_uncertain_topology_is_retained_without_positional_mapping(table, text):
    for cell in table.cells:
        cell.raw_text = None
    page = page_for([table], text)
    attach_page_reading(page)
    assert page.issues[0].code == "page_alignment_uncertain"
    assert all(c.raw_text is None for c in table.cells)
    assert page.annotated_reading.text == text


def test_multiple_tables_with_identical_topology_are_not_guessed(table):
    other = table.model_copy(deep=True)
    other.id = "p1-t2"
    text = html_for(table) + html_for(other)
    for target in (table, other):
        for cell in target.cells:
            cell.raw_text = None
    page = page_for([table, other], text)
    attach_page_reading(page)
    assert page.issues and all(c.raw_text is None for t in page.tables for c in t.cells)
