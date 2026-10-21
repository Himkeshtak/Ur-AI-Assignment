"""Bind readings to fixed cell IDs and interpret conservative invoice layouts."""

import re
import unicodedata
from html.parser import HTMLParser

from .models import InvoiceLine, InvoiceView, Issue, Page, Table

ALIASES = {
    "quantity": {"数量", "個数", "qty", "quantity"},
    "unit_price": {"単価", "unitprice", "price"},
    "amount": {"金額", "明細金額", "amount", "lineamount"},
}
TOTAL_LABELS = {
    "subtotal": {"小計", "subtotal"},
    "total": {"合計", "総合計", "請求金額", "税込合計", "total", "grandtotal"},
    "tax": {"消費税", "税額", "tax"},
    "adjustment": {"値引", "値引き", "割引", "送料", "調整額", "discount", "shipping"},
}


def normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value)).casefold()


def map_invoice(table: Table) -> InvoiceView | None:
    by_position = {(c.row, c.column): c for c in table.cells if c.column_span == 1}
    choices = []
    for row in table.rows[:5]:
        mapping: dict[str, int] = {}
        ambiguous = False
        for cell in table.cells:
            if cell.row != row.index or cell.column_span != 1:
                continue
            for role, aliases in ALIASES.items():
                if normalized(cell.raw_text or "") in aliases:
                    if role in mapping:
                        ambiguous = True
                    mapping[role] = cell.column
        if "amount" in mapping and not ambiguous:
            choices.append((len(mapping), row.index, mapping))
    if not choices:
        return None
    _, header_row, columns = max(choices, key=lambda item: (item[0], -item[1]))
    view = InvoiceView(header_row=header_row, columns=columns)
    for row in table.rows:
        if row.index <= header_row:
            continue
        cells = [c for c in table.cells if c.row == row.index]
        labels = {normalized(c.raw_text or "") for c in cells}
        kind = next((kind for kind, aliases in TOTAL_LABELS.items() if labels & aliases), None)
        amount = by_position.get((row.index, columns["amount"]))
        if kind:
            if amount is None or normalized(amount.raw_text or "") in TOTAL_LABELS[kind]:
                table.issues.append(
                    Issue(
                        code="unmapped_total",
                        stage="J",
                        message=f"Could not identify {kind} amount",
                    )
                )
                continue
            if kind in {"subtotal", "total"}:
                field = f"{kind}_id"
                if getattr(view, field) is not None:
                    table.issues.append(
                        Issue(
                            code="duplicate_total",
                            stage="J",
                            message=f"Multiple {kind} rows need explicit mapping",
                        )
                    )
                else:
                    setattr(view, field, amount.id)
            elif kind == "tax":
                view.tax_ids.append(amount.id)
            else:
                view.adjustment_ids.append(amount.id)
            continue
        if not any(c.raw_text and c.raw_text.strip() for c in cells):
            continue
        if labels & ALIASES["amount"]:  # Repeated header, not a line item.
            continue
        if amount is None:
            table.issues.append(
                Issue(
                    code="unmapped_line",
                    stage="J",
                    message=f"Row {row.index} has no unique amount cell",
                )
            )
            continue
        related = {
            role: by_position.get((row.index, columns[role]))
            for role in ("quantity", "unit_price")
            if role in columns
        }
        view.lines.append(
            InvoiceLine(
                row=row.index,
                amount_id=amount.id,
                quantity_id=related["quantity"].id if related.get("quantity") else None,
                unit_price_id=related["unit_price"].id if related.get("unit_price") else None,
            )
        )
    return view


class _TableHTML(HTMLParser):
    """Read explicit cell spans; reject malformed or nested table ownership."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.tables: list[dict[tuple[int, int, int, int], str]] = []
        self.cells: dict[tuple[int, int, int, int], str] | None = None
        self.occupied: set[tuple[int, int]] = set()
        self.row = 0
        self.in_row = False
        self.active: tuple[int, int, int, int] | None = None
        self.parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag == "table":
            if self.cells is not None:
                raise ValueError("Nested tables cannot be mapped to the grid")
            self.cells, self.occupied, self.row = {}, set(), 0
        elif tag == "tr" and self.cells is not None:
            if self.in_row:
                raise ValueError("Unclosed table row")
            self.row += 1
            self.in_row = True
        elif tag in {"td", "th"} and self.cells is not None:
            if self.active or not self.in_row:
                raise ValueError("Malformed table cell")
            attributes = dict(attrs)
            rs, cs = int(attributes.get("rowspan", "1")), int(attributes.get("colspan", "1"))
            if not 1 <= rs <= 500 or not 1 <= cs <= 500:
                raise ValueError("Unsupported HTML span")
            column = 1
            while (self.row, column) in self.occupied:
                column += 1
            covered = {
                (r, c) for r in range(self.row, self.row + rs) for c in range(column, column + cs)
            }
            if covered & self.occupied:
                raise ValueError("HTML cell ownership overlaps")
            self.occupied.update(covered)
            self.active, self.parts = (self.row, column, rs, cs), []
        elif tag == "br" and self.active:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.active:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag in {"td", "th"} and self.cells is not None:
            if self.active is None:
                raise ValueError("Unexpected cell end")
            self.cells[self.active] = "".join(self.parts)
            self.active = None
        elif tag == "tr" and self.cells is not None:
            if self.active or not self.in_row:
                raise ValueError("Unclosed table cell")
            self.in_row = False
        elif tag == "table":
            if self.cells is None or self.active or self.in_row:
                raise ValueError("Unclosed table row or unexpected table end")
            if any(r > self.row for r, _ in self.occupied):
                raise ValueError("Row span extends beyond the table")
            self.tables.append(self.cells)
            self.cells = None


def attach_page_reading(page: Page) -> None:
    """Bind one page response only through an exact, unique grid topology match.

    Paddle's table endpoint returns OTSL/HTML, not grounded cell IDs or boxes.
    A mismatch is retained as evidence instead of shifting text into nearby cells.
    """
    reading = page.annotated_reading
    try:
        if reading is None or reading.error or not reading.text:
            raise ValueError(
                reading.error if reading and reading.error else "Page OCR is incomplete"
            )
        parser = _TableHTML()
        parser.feed(reading.text)
        parser.close()
        if parser.cells is not None or not parser.tables:
            raise ValueError("OCR did not return complete table HTML")
        if len(parser.tables) != len(page.tables):
            raise ValueError("OCR table count differs from the detected table count")
        if any(not t.geometry_valid for t in page.tables):
            raise ValueError(
                "TATR geometry is unresolved; OCR tables were retained without assignment"
            )
        grids = {
            t.id: {(c.row, c.column, c.row_span, c.column_span): c for c in t.cells}
            for t in page.tables
        }
        matches = []
        for parsed in parser.tables:
            targets = [t for t in page.tables if grids[t.id].keys() == parsed.keys()]
            if len(targets) != 1:
                raise ValueError(
                    "OCR topology differs from the fixed grid or matches multiple tables"
                )
            matches.append((targets[0], parsed))
        if len({t.id for t, _ in matches}) != len(page.tables):
            raise ValueError("OCR tables cannot be assigned uniquely")
    except (ValueError, KeyError) as exc:
        page.issues.append(Issue(code="page_alignment_uncertain", stage="J", message=str(exc)))
        return
    for table, parsed in matches:
        for position, cell in grids[table.id].items():
            cell.raw_text = parsed[position]
            cell.normalized_text = unicodedata.normalize("NFKC", cell.raw_text).strip()
            cell.readings.append(
                reading.model_copy(
                    update={
                        "text": cell.raw_text,
                        "source_bbox_in_image": cell.bbox,
                    }
                )
            )
        table.invoice = map_invoice(table)
