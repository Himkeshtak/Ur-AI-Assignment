import pytest
from conftest import structure_objects
from PIL import Image

from jp_pdf_parser.grid import (
    annotated_page,
    reconstruct_grid,
    transform_box,
)
from jp_pdf_parser.models import Detection
from jp_pdf_parser.validation import validate_geometry


def test_grid_preserves_row_membership_and_merges(table):
    assert table.geometry_valid
    assert len(table.cells) == 12
    amount = next(c for c in table.cells if (c.row, c.column) == (2, 4))
    assert amount.bbox == (350, 90, 450, 130)
    assert amount.row_ids == ["p1-t1-g1-r2"]
    assert amount.column_ids == ["p1-t1-g1-c4"]
    subtotal = next(c for c in table.cells if (c.row, c.column) == (3, 1))
    assert subtotal.column_span == 3
    assert len(subtotal.column_ids) == 3


def test_empty_cells_are_not_removed(table):
    table.cells[0].raw_text = ""
    assert len(table.cells) == 12


def test_missing_axes_and_cell_limits_fail_closed():
    invalid = reconstruct_grid("p1-t1", 1, "x.png", (424, 184), (1, 0, 0, 0, 1, 0), [], 0.6, 500)
    assert not invalid.geometry_valid
    limited = reconstruct_grid(
        "p1-t1", 1, "x.png", (424, 184), (1, 0, 0, 0, 1, 0), structure_objects(), 0.6, 5
    )
    assert limited.issues[0].code == "cell_limit"


def test_full_page_overlay_preserves_dimensions_coordinates_and_source(table):
    with Image.new("RGB", (500, 300), "white") as image:
        original = image.tobytes()
        # The runtime uses page coordinates with an identity transform.
        for axis in [*table.rows, *table.columns]:
            axis.bbox = transform_box(axis.bbox, table.table_to_page)
        with annotated_page(image, [table], 10_000_000) as overlay:
            assert overlay.size == image.size
            assert overlay.getpixel((38, 38)) == (0, 128, 0)
        assert image.tobytes() == original
        with pytest.raises(ValueError, match="pixel limit"):
            annotated_page(image, [table], 100)


def test_page_grid_limits_axes_to_detected_table():
    result = reconstruct_grid(
        "p1-t1",
        1,
        "page.png",
        (500, 300),
        (1, 0, 0, 0, 1, 0),
        structure_objects(),
        0.6,
        500,
        region=(12, 12, 412, 172),
    )
    assert result.geometry_valid
    assert result.bbox == (12, 12, 412, 172)
    assert result.structure_coordinate_frame == "rendered_page_pixels"
    assert all(a.coordinate_frame == "rendered_page_pixels" for a in result.rows + result.columns)
    assert all(c.bbox == c.table_bbox for c in result.cells)


def test_conflicting_spans_require_review():
    objects = [o for o in structure_objects() if o.label != "table spanning cell"]
    objects += [
        Detection(label="table spanning cell", score=0.98, bbox=box)
        for box in (
            (12, 92, 212, 132),
            (112, 92, 312, 132),
        )
    ]
    result = reconstruct_grid(
        "p1-t1", 1, "x.png", (424, 184), (1, 0, 0, 0, 1, 0), objects, 0.6, 500
    )
    assert not result.geometry_valid
    assert any(issue.code == "conflicting_span" for issue in result.issues)


def test_missing_grid_position_is_detected(table):
    table.cells.pop()
    assert any(issue.code == "grid_coverage" for issue in validate_geometry(table))
