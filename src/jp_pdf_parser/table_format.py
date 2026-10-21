"""Strict OTSL-to-HTML adaptation for the annotated-table comparison only."""

import re
from html import escape

TOKEN = re.compile(r"<(fcel|ecel|lcel|ucel|xcel|nl)>")


def otsl_to_html(text: str) -> str:
    """Reject ambiguous spans instead of inventing row/column ownership."""
    if text.lstrip().startswith("<table"):
        return text  # The comparison layer independently checks the HTML topology.
    matches = list(TOKEN.finditer(text))
    if not matches or text[: matches[0].start()].strip():
        raise ValueError("Expected an OTSL table without a prose prefix")
    rows: list[list[tuple[str, str]]] = []
    row = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        tag, content = match.group(1), text[match.end() : end]
        if tag == "nl":
            if content.strip() or not row:
                raise ValueError("Invalid OTSL row delimiter")
            rows.append(row)
            row = []
        else:
            if tag != "fcel" and content.strip():
                raise ValueError("OTSL continuation/empty cells cannot contain text")
            row.append((tag, content if tag == "fcel" else ""))
    if row:
        rows.append(row)
    if not rows or len({len(row) for row in rows}) != 1:
        raise ValueError("OTSL rows must have equal widths")
    if len(rows) * len(rows[0]) > 10000:
        raise ValueError("OTSL table exceeds the supported slot limit")
    owners: dict[tuple[int, int], tuple[int, int]] = {}
    cells: dict[tuple[int, int], str] = {}
    slots: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, row in enumerate(rows):
        for c, (tag, content) in enumerate(row):
            position = r, c
            if tag in {"fcel", "ecel"}:
                owner = position
                cells[owner] = content
                slots[owner] = []
            elif tag == "lcel":
                owner = owners.get((r, c - 1))
                if owner is None or owner[0] != r:
                    raise ValueError("OTSL horizontal continuation has no matching cell")
            elif tag == "ucel":
                owner = owners.get((r - 1, c))
                if owner is None or owner[1] != c:
                    raise ValueError("OTSL vertical continuation has no matching cell")
            else:  # xcel extends the rectangle both horizontally and vertically.
                owner = owners.get((r - 1, c))
                if owner is None or owner != owners.get((r, c - 1)):
                    raise ValueError("OTSL cross continuation has conflicting owners")
                if owner[0] >= r or owner[1] >= c:
                    raise ValueError("OTSL cross continuation is not inside a merged cell")
            owners[position] = owner
            slots[owner].append(position)
    spans = {}
    for (r, c), positions in slots.items():
        bottom = max(p[0] for p in positions)
        right = max(p[1] for p in positions)
        if set(positions) != {(y, x) for y in range(r, bottom + 1) for x in range(c, right + 1)}:
            raise ValueError("OTSL merged cell is not rectangular")
        spans[(r, c)] = bottom - r + 1, right - c + 1
    output = ["<table>"]
    for r in range(len(rows)):
        output.append("<tr>")
        for c in range(len(rows[0])):
            if (r, c) in cells:
                rs, cs = spans[(r, c)]
                value = escape(cells[(r, c)]).replace("\n", "<br/>")
                output.append(f'<td rowspan="{rs}" colspan="{cs}">{value}</td>')
        output.append("</tr>")
    output.append("</table>")
    return "".join(output)
