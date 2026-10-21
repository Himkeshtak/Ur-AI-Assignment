import pytest

from jp_pdf_parser.table_format import otsl_to_html
from jp_pdf_parser.tables import _TableHTML


def parsed(text):
    parser = _TableHTML()
    parser.feed(otsl_to_html(text))
    parser.close()
    return parser.tables[0]


def test_otsl_preserves_numbers_empty_cells_and_escapes_html():
    assert parsed("<fcel>A & B<fcel>2,000<nl><ecel><fcel><script>x</script><nl>") == {
        (1, 1, 1, 1): "A & B",
        (1, 2, 1, 1): "2,000",
        (2, 1, 1, 1): "",
        (2, 2, 1, 1): "<script>x</script>",
    }


def test_otsl_reconstructs_rectangular_spans_and_continuations():
    assert parsed("<fcel>merged<lcel><fcel>top<nl><ucel><xcel><fcel>bottom<nl>") == {
        (1, 1, 2, 2): "merged",
        (1, 3, 1, 1): "top",
        (2, 3, 1, 1): "bottom",
    }


def test_vertical_and_horizontal_spans_are_distinct():
    assert parsed("<fcel>A<fcel>B<lcel><nl><ucel><fcel>C<fcel>D<nl>") == {
        (1, 1, 2, 1): "A",
        (1, 2, 1, 2): "B",
        (2, 2, 1, 1): "C",
        (2, 3, 1, 1): "D",
    }


@pytest.mark.parametrize(
    "text",
    [
        "",
        "explanation<fcel>A<nl>",
        "<lcel><nl>",
        "<ucel><nl>",
        "<xcel><nl>",
        "<ecel>text<nl>",
        "<fcel>A<nl><fcel>B<fcel>C<nl>",
        "<nl>",
        "<fcel>A<lcel><nl><ucel><fcel>break rectangle<nl>",
        "<fcel>A<fcel>B<nl><ucel><xcel><nl>",
    ],
)
def test_malformed_or_ambiguous_tables_are_rejected(text):
    with pytest.raises(ValueError):
        otsl_to_html(text)
