from decimal import Decimal

import pytest

from jp_pdf_parser.validation import NumberError, parse_number


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2,000", "2000"),
        ("２，０００", "2000"),
        ("￥2,000", "2000"),
        ("JPY 2,000", "2000"),
        ("2,000円", "2000"),
        ("(2,000)", "-2000"),
        ("▲2,000", "-2000"),
        ("−2,000", "-2000"),
        ("2.50", "2.50"),
        ("0", "0"),
    ],
)
def test_number_formats(raw, expected):
    assert parse_number(raw, monetary=True) == Decimal(expected)


@pytest.mark.parametrize("raw", ["20,00", "2.000", "2,O00", "2 000", "1e3", "NaN", "", "(-20)"])
def test_ambiguous_and_invalid_amounts_are_rejected(raw):
    with pytest.raises(NumberError):
        parse_number(raw, monetary=True)


def test_fractional_quantities_are_allowed():
    assert parse_number("0.125") == Decimal("0.125")


def test_raw_value_is_never_modified():
    raw = "２，０００円"
    parse_number(raw, monetary=True)
    assert raw == "２，０００円"
