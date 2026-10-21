from decimal import Decimal

from jp_pdf_parser.config import FinancialSettings
from jp_pdf_parser.validation import validate_invoice


def test_line_subtotal_and_total_reconcile(table):
    checks = validate_invoice(table, FinancialSettings())
    assert [c.rule for c in checks] == ["line_amount", "subtotal", "total"]
    assert all(c.status == "passed" for c in checks)
    assert next(c for c in table.cells if c.raw_text == "1,000").number == "1000"


def test_balanced_subtotal_does_not_hide_a_bad_line(table):
    next(c for c in table.cells if (c.row, c.column) == (2, 2)).raw_text = "3"
    checks = validate_invoice(table, FinancialSettings())
    assert checks[0].status == "failed"
    assert checks[0].difference == "-1000"
    assert checks[-1].status == "passed"


def test_decimal_separator_corruption_does_not_become_2000(table):
    cell = next(c for c in table.cells if (c.row, c.column) == (2, 4))
    cell.raw_text = "2.000"
    checks = validate_invoice(table, FinancialSettings())
    assert cell.number is None
    assert any(i.code == "invalid_number" and cell.id in i.cell_ids for i in table.issues)
    assert any(c.status == "incomplete" for c in checks)


def test_unknown_tax_policy_is_incomplete(table):
    next(c for c in table.cells if c.raw_text == "小計").raw_text = "消費税"
    checks = validate_invoice(table, FinancialSettings())
    assert checks[-1].status == "incomplete"
    assert "policy" in checks[-1].detail


def test_explicit_exclusive_tax_policy(table):
    next(c for c in table.cells if c.raw_text == "小計").raw_text = "消費税"
    next(c for c in table.cells if (c.row, c.column) == (3, 4)).raw_text = "200"
    next(c for c in table.cells if (c.row, c.column) == (4, 4)).raw_text = "2,200"
    settings = FinancialSettings(
        tax_mode="exclusive", tax_rate=Decimal("0.10"), rounding="ROUND_HALF_UP"
    )
    assert all(c.status == "passed" for c in validate_invoice(table, settings))


def test_rounding_is_only_used_when_configured(table):
    next(c for c in table.cells if (c.row, c.column) == (2, 2)).raw_text = "0.3333"
    next(c for c in table.cells if (c.row, c.column) == (2, 3)).raw_text = "6000"
    assert validate_invoice(table, FinancialSettings())[0].status == "failed"
    assert (
        validate_invoice(table, FinancialSettings(rounding="ROUND_HALF_UP"))[0].status == "passed"
    )


def test_unknown_headers_require_review_in_invoice_mode(table):
    next(c for c in table.cells if c.raw_text == "金額").raw_text = "不明"
    assert validate_invoice(table, FinancialSettings())[0].status == "incomplete"
    assert validate_invoice(table, FinancialSettings(mode="table"))[0].status == "not_applicable"


def test_large_amounts_keep_exact_precision(table):
    value = "12345678901234567890123456789012"
    next(c for c in table.cells if (c.row, c.column) == (2, 2)).raw_text = "1"
    next(c for c in table.cells if (c.row, c.column) == (2, 3)).raw_text = value
    for cell in table.cells:
        if cell.column == 4 and cell.row > 1:
            cell.raw_text = value
    assert all(check.status == "passed" for check in validate_invoice(table, FinancialSettings()))
