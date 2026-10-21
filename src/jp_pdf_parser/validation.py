"""Strict Japanese invoice number parsing and explainable Decimal checks."""

import re
import unicodedata
from decimal import Decimal, localcontext

from .config import FinancialSettings
from .grid import transform_box
from .models import Check, Document, Issue, Table
from .tables import map_invoice


class NumberError(ValueError):
    pass


def parse_number(raw: str, *, monetary: bool = False) -> Decimal:
    text = unicodedata.normalize("NFKC", raw).strip().replace("−", "-")
    negative = text.startswith(("(", "▲", "△"))
    if text.startswith("("):
        if not text.endswith(")"):
            raise NumberError("Unbalanced negative amount parentheses")
        text = text[1:-1]
    elif text.startswith(("▲", "△")):
        text = text[1:]
    text = re.sub(r"^(?:JPY\s*|¥\s*)", "", text, flags=re.I)
    text = re.sub(r"\s*(?:円|JPY)$", "", text, flags=re.I).strip()
    if len(text) > 64:
        raise NumberError("Number is longer than the supported decimal precision")
    if not re.fullmatch(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", text):
        raise NumberError("Invalid number or thousands grouping")
    if negative and text.startswith(("-", "+")):
        raise NumberError("Conflicting amount signs")
    if monetary and re.search(r"\.\d{3}$", text):
        raise NumberError("Ambiguous decimal point versus a thousands separator")
    value = Decimal(text.replace(",", ""))
    return -value if negative else value


def _comparison(
    rule: str, expected: Decimal, observed: Decimal, ids: list[str], detail: str
) -> Check:
    return Check(
        rule=rule,
        status="passed" if expected == observed else "failed",
        cell_ids=ids,
        expected=str(expected),
        observed=str(observed),
        difference=str(observed - expected),
        detail=detail,
    )


def validate_invoice(table: Table, settings: FinancialSettings) -> list[Check]:
    if settings.mode == "table":
        return [
            Check(
                rule="invoice_mapping",
                status="not_applicable",
                detail="Generic-table mode was explicitly selected",
            )
        ]
    view = table.invoice or map_invoice(table)
    table.invoice = view
    if view is None:
        return [
            Check(
                rule="invoice_mapping",
                status="incomplete",
                detail="No unambiguous amount header was identified",
            )
        ]
    cells = {cell.id: cell for cell in table.cells}
    checks: list[Check] = []
    values: dict[str, Decimal] = {}
    quantity_ids = {line.quantity_id for line in view.lines}
    unit_price_ids = {line.unit_price_id for line in view.lines}
    numeric_ids = (
        {line.amount_id for line in view.lines}
        | quantity_ids
        | unit_price_ids
        | {
            view.subtotal_id,
            view.total_id,
            *view.tax_ids,
            *view.adjustment_ids,
        }
    )
    for cell_id in sorted(numeric_ids - {None}):
        cell = cells[cell_id]
        try:
            value = parse_number(cell.raw_text or "", monetary=cell_id not in quantity_ids)
            if cell_id not in quantity_ids | unit_price_ids:
                quantum = Decimal(1).scaleb(-settings.money_scale)
                with localcontext() as context:
                    context.prec = 128
                    if value % quantum != 0:
                        raise NumberError("Amount has unsupported fractional currency units")
            values[cell_id] = value
            cell.number = str(value)
        except NumberError as exc:
            table.issues.append(
                Issue(code="invalid_number", stage="K", cell_ids=[cell_id], message=str(exc))
            )
    with localcontext() as context:
        context.prec = 128
        for line in view.lines:
            ids = [line.quantity_id, line.unit_price_id, line.amount_id]
            if "quantity" not in view.columns or "unit_price" not in view.columns:
                continue
            if any(cell_id not in values for cell_id in ids):
                checks.append(
                    Check(
                        rule="line_amount",
                        status="incomplete",
                        cell_ids=[i for i in ids if i],
                        detail="Quantity, unit price, or line amount is missing/invalid",
                    )
                )
                continue
            expected = values[line.quantity_id] * values[line.unit_price_id]
            if settings.rounding:
                expected = expected.quantize(
                    Decimal(1).scaleb(-settings.money_scale), rounding=settings.rounding
                )
            checks.append(
                _comparison(
                    "line_amount",
                    expected,
                    values[line.amount_id],
                    ids,
                    "quantity * unit_price; configured line rounding if set",
                )
            )
        amount_ids = [line.amount_id for line in view.lines]
        if not amount_ids or any(i not in values for i in amount_ids):
            checks.append(
                Check(
                    rule="subtotal",
                    status="incomplete",
                    detail="Line amounts are missing or invalid",
                )
            )
            return checks
        line_sum = sum((values[i] for i in amount_ids), Decimal(0))
        if view.subtotal_id:
            if view.subtotal_id not in values:
                checks.append(
                    Check(
                        rule="subtotal",
                        status="incomplete",
                        detail="Recognized subtotal is invalid",
                    )
                )
                return checks
            checks.append(
                _comparison(
                    "subtotal",
                    line_sum,
                    values[view.subtotal_id],
                    [*amount_ids, view.subtotal_id],
                    "sum(line amounts)",
                )
            )
        if view.total_id not in values:
            checks.append(
                Check(
                    rule="total",
                    status="incomplete",
                    detail="A readable total in the detected table is required",
                )
            )
            return checks
        extras = [*view.tax_ids, *view.adjustment_ids]
        if any(i not in values for i in extras):
            checks.append(
                Check(
                    rule="total", status="incomplete", detail="Tax or adjustment amount is invalid"
                )
            )
            return checks
        if view.adjustment_ids:
            checks.append(
                Check(
                    rule="total",
                    status="incomplete",
                    cell_ids=view.adjustment_ids,
                    detail="Adjustment semantics require an explicit invoice profile",
                )
            )
            return checks
        if view.tax_ids and settings.tax_mode == "unknown":
            checks.append(
                Check(
                    rule="total",
                    status="incomplete",
                    cell_ids=view.tax_ids,
                    detail="Tax-inclusive/exclusive policy is unspecified",
                )
            )
            return checks
        expected = line_sum
        if settings.tax_mode == "exclusive":
            expected += sum((values[i] for i in view.tax_ids), Decimal(0))
        if view.tax_ids:
            if len(view.tax_ids) == 1 and settings.tax_rate is not None and settings.rounding:
                tax = line_sum * settings.tax_rate
                if settings.tax_mode == "inclusive":
                    tax = line_sum * settings.tax_rate / (1 + settings.tax_rate)
                tax = tax.quantize(
                    Decimal(1).scaleb(-settings.money_scale), rounding=settings.rounding
                )
                checks.append(
                    _comparison(
                        "tax",
                        tax,
                        values[view.tax_ids[0]],
                        view.tax_ids,
                        "Configured rate, tax mode, and rounding",
                    )
                )
            else:
                checks.append(
                    Check(
                        rule="tax",
                        status="incomplete",
                        cell_ids=view.tax_ids,
                        detail="Tax bases/rates/rounding need an explicit profile",
                    )
                )
        checks.append(
            _comparison(
                "total",
                expected,
                values[view.total_id],
                [*amount_ids, *extras, view.total_id],
                "line sum plus exclusive tax, or tax-inclusive line sum",
            )
        )
    return checks


def validate_geometry(table: Table) -> list[Issue]:
    expected = {(r.index, c.index) for r in table.rows for c in table.columns}
    occupied: set[tuple[int, int]] = set()
    issues = []
    ids = [cell.id for cell in table.cells]
    if len(ids) != len(set(ids)):
        issues.append(
            Issue(
                code="duplicate_cell_id",
                stage="K",
                message="Cell IDs must be unique within a grid",
                severity="error",
            )
        )
    for cell in table.cells:
        positions = {
            (r, c)
            for r in range(cell.row, cell.row + cell.row_span)
            for c in range(cell.column, cell.column + cell.column_span)
        }
        rows = [r.id for r in table.rows if cell.row <= r.index < cell.row + cell.row_span]
        columns = [
            c.id for c in table.columns if cell.column <= c.index < cell.column + cell.column_span
        ]
        if (
            positions - expected
            or positions & occupied
            or rows != cell.row_ids
            or columns != cell.column_ids
        ):
            issues.append(
                Issue(
                    code="cell_ownership",
                    stage="K",
                    cell_ids=[cell.id],
                    message="Cell span or row/column ownership is inconsistent",
                    severity="error",
                )
            )
        occupied.update(positions)
        projected = transform_box(cell.table_bbox, table.table_to_page)
        if any(abs(a - b) > 1e-6 for a, b in zip(projected, cell.bbox, strict=True)):
            issues.append(
                Issue(
                    code="coordinate_mismatch",
                    stage="K",
                    cell_ids=[cell.id],
                    message="Cell page coordinates disagree with its table transform",
                    severity="error",
                )
            )
    if occupied != expected:
        issues.append(
            Issue(
                code="grid_coverage",
                stage="K",
                severity="error",
                message="The reconstructed grid has missing or excess positions",
            )
        )
    return issues


def validate_document(document: Document, settings: FinancialSettings) -> None:
    for page in document.pages:
        if not page.tables:
            page.issues.append(
                Issue(code="no_tables", stage="K", message="No tables were detected on this page")
            )
        for table in page.tables:
            if not table.geometry_valid:
                continue
            geometry_issues = validate_geometry(table)
            if geometry_issues:
                table.issues.extend(geometry_issues)
                table.geometry_valid = False
                continue
            for cell in table.cells:
                if cell.raw_text is None or not cell.raw_text.strip():
                    table.issues.append(
                        Issue(
                            code="empty_or_unreadable",
                            stage="K",
                            cell_ids=[cell.id],
                            message="Empty OCR does not establish a blank cell",
                        )
                    )
                elif "\ufffd" in cell.raw_text or any(
                    unicodedata.category(char) == "Cc" and char not in "\n\r\t"
                    for char in cell.raw_text
                ):
                    table.issues.append(
                        Issue(
                            code="suspicious_encoding",
                            stage="K",
                            cell_ids=[cell.id],
                            message="Text contains replacement/control characters",
                        )
                    )
            document.checks.extend(validate_invoice(table, settings))


def final_status(document: Document) -> str:
    readable = any(cell.raw_text and cell.raw_text.strip() for cell in document.all_cells())
    readable = readable or any(
        page.annotated_reading
        and not page.annotated_reading.error
        and page.annotated_reading.text
        and page.annotated_reading.text.strip()
        for page in document.pages
    )
    if not readable:
        fatal = any(
            issue.severity == "error" or issue.code == "cell_ocr_failed"
            for issue in document.all_issues()
        )
        return "failed" if fatal or not document.pages else "needs_review"
    complete_pages = len(document.pages) == document.page_count and document.page_count > 0
    complete_review = document.japanese_review is not None and document.japanese_review.completed
    unresolved = document.all_issues() or any(
        check.status in {"failed", "incomplete"} for check in document.checks
    )
    if not complete_pages or not complete_review or unresolved:
        return "needs_review"
    return "checks_passed"
