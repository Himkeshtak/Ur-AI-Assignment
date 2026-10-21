from jp_pdf_parser.models import Reading
from jp_pdf_parser.tables import compare_annotated_reading, map_invoice


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


def test_annotated_table_agreement(table):
    table.annotated_reading = Reading(
        request_id="x", variant="annotated_table", image_path="x.png", text=html_for(table)
    )
    compare_annotated_reading(table)
    assert not table.issues


def test_overlay_digits_cannot_replace_clean_amounts(table):
    table.annotated_reading = Reading(
        request_id="x",
        variant="annotated_table",
        image_path="x.png",
        text=html_for(table).replace("2,000", "R2 C4 2,000"),
    )
    compare_annotated_reading(table)
    assert any(i.code == "annotated_disagreement" for i in table.issues)
    assert next(c for c in table.cells if (c.row, c.column) == (2, 4)).raw_text == "2,000"


def test_different_annotated_topology_is_not_matched_by_position(table):
    table.annotated_reading = Reading(
        request_id="x",
        variant="annotated_table",
        image_path="x.png",
        text="<table><tr><td>2,000</td></tr></table>",
    )
    compare_annotated_reading(table)
    assert table.issues[-1].code == "annotated_alignment_uncertain"
