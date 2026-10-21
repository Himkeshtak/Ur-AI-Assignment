import pytest
from conftest import structure_objects
from PIL import Image

from jp_pdf_parser.grid import (
    cell_crop,
    cell_pixel_box,
    numbered_overlay,
    reconstruct_grid,
    transform_box,
)
from jp_pdf_parser.models import Detection
from jp_pdf_parser.pdf import crop_table
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


def test_rotated_crop_coordinate_transform():
    with Image.new("RGB", (200, 150), "white") as image:
        detection = Detection(label="table rotated", score=1, bbox=(20, 30, 120, 90))
        crop, transform = crop_table(image, detection, 0)
        with crop:
            assert crop.size == (60, 100)
            assert transform_box((0, 0, 60, 100), transform) == (20, 30, 120, 90)
            assert transform_box((0, 0, 10, 20), transform) == (20, 80, 40, 90)


def test_overlay_labels_do_not_change_source_or_clean_crops(table):
    with Image.new("RGB", (424, 184), "white") as image:
        original = image.tobytes()
        with numbered_overlay(image, table, 10_000_000) as overlay:
            assert overlay.width > image.width
        assert image.tobytes() == original
        with cell_crop(image, table.cells[0]) as crop:
            assert crop.size == (100, 40)
        with pytest.raises(ValueError, match="pixel limit"):
            numbered_overlay(image, table, 100)


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


def test_fractional_shared_boundaries_do_not_overlap(table):
    left, right = table.cells[0], table.cells[1]
    left.table_bbox = (12, 12, 112.4, 52)
    right.table_bbox = (112.4, 12, 212, 52)
    with Image.new("RGB", (424, 184)) as image:
        assert cell_pixel_box(image, left)[2] == cell_pixel_box(image, right)[0]
