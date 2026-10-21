import pytest

from jp_pdf_parser.artifacts import Artifacts
from jp_pdf_parser.grid import transform_box
from jp_pdf_parser.pdf import inspect_pdf, render_pages


def test_real_pdf_render_and_coordinate_mapping(config, pdf, tmp_path):
    digest, count = inspect_pdf(pdf, config.render)
    assert len(digest) == 64 and count == 1
    pages = list(render_pages(pdf, config.render, Artifacts(tmp_path / "output")))
    page = pages[0]
    assert (page.width_pixels, page.height_pixels) == (500, 300)
    assert transform_box((0, 0, 500, 300), page.image_to_pdf) == pytest.approx((0, 0, 500, 300))


def test_page_pixel_limit_is_enforced_before_render(config, pdf, tmp_path):
    config.render.max_pixels = 1000
    with pytest.raises(ValueError, match="pixel limit"):
        list(render_pages(pdf, config.render, Artifacts(tmp_path / "output")))


def test_non_pdf_input_is_rejected(config, tmp_path):
    source = tmp_path / "not-a-pdf.pdf"
    source.write_text("not a pdf")
    with pytest.raises(ValueError, match="signature"):
        inspect_pdf(source, config.render)
