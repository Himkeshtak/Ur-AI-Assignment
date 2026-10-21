"""Local-only spot checks and a safe HTML preview; sends no model requests."""

import argparse
import json
import re
from html import escape
from pathlib import Path

from experiments.no_tatr import ExperimentDocument, production_hashes
from jp_pdf_parser.artifacts import sha256, write_json
from jp_pdf_parser.validation import NumberError, parse_number

# Manually read from the source PDF, not inferred from model output.
# Values are in the document's printed units (thousands of yen).
REFERENCES = {
    "現金及び預金": ("2002429", "608281"),
    "流動資産合計": ("5863684", "3377407"),
    "固定資産合計": ("1670426", "1757841"),
    "資産合計": ("7534110", "5135249"),
}


def evaluate(directory: Path) -> Path:
    document = ExperimentDocument.model_validate_json(
        (directory / "result.json").read_text(encoding="utf-8")
    )
    config = json.loads((directory / "experiment.json").read_text(encoding="utf-8"))
    assert production_hashes() == config["production_sha256"]
    checks = []
    for label, amounts in REFERENCES.items():
        labels = [c for c in document.all_cells() if c.column == 1 and c.raw_text == label]
        for column, expected in enumerate(amounts, 2):
            candidates = [
                c
                for p in document.pages
                for t in p.tables
                for c in t.cells
                if len(labels) == 1
                and labels[0] in t.cells
                and c.row == labels[0].row
                and c.column == column
            ]
            cell = candidates[0] if len(candidates) == 1 else None
            raw = cell.raw_text if cell else None
            observed, valid_format = None, False
            if raw is not None:
                try:
                    observed = str(parse_number(raw, monetary=True))
                    valid_format = True
                except NumberError:
                    try:
                        # Ignore whitespace only for this numeric spot check; retain punctuation.
                        observed = str(parse_number(re.sub(r"\s+", "", raw), monetary=True))
                    except NumberError:
                        pass
            checks.append(
                {
                    "label": label,
                    "column": column,
                    "cell_id": cell.id if cell else None,
                    "raw_text": raw,
                    "expected": expected,
                    "observed": observed,
                    "numeric_match": observed == expected,
                    "raw_format_valid": valid_format,
                }
            )
    traces = json.loads((directory / "japanese-traces.json").read_text(encoding="utf-8"))
    returned_ids = []
    if traces and traces[0].get("answer"):
        returned_ids = json.loads(traces[0]["answer"]).get("reviewed_cell_ids", [])
    report = {
        "reference_method": "Manual visual spot checks on four source rows; not full ground truth",
        "source_name": document.source_name,
        "source_sha256": document.source_sha256,
        "numeric_spot_checks": checks,
        "matched": sum(c["numeric_match"] for c in checks),
        "checked": len(checks),
        "raw_text_unchanged": True,
        "paddle_layout_issue": (
            "Correction note and unit label were included as an extra first table row"
        ),
        "japanese_requested_ids": len(document.all_cells()),
        "japanese_returned_ids": len(returned_ids),
        "japanese_completed": document.japanese_review.completed,
        "production_unchanged": True,
    }
    write_json(directory / "evaluation.json", report)
    preview = [
        '<!doctype html><html lang="ja"><meta charset="utf-8">',
        "<title>No-TATR experiment</title>",
        "<style>body{font-family:system-ui,sans-serif;margin:2rem}"
        "table{border-collapse:collapse;margin-bottom:2rem}"
        "td{border:1px solid #aaa;padding:.35rem .7rem;white-space:pre-wrap}</style>",
        "<h1>No-TATR: raw logical table</h1>",
        "<p>Needs review. No pixel coordinates. "
        "Source text is preserved, including OCR spacing.</p>",
    ]
    for page in document.pages:
        for table in page.tables:
            preview.append(f"<h2>{escape(table.id)}</h2><table>")
            for row in table.rows:
                preview.append("<tr>")
                for cell in table.cells:
                    if cell.row == row.index:
                        preview.append(
                            f'<td rowspan="{cell.row_span}" colspan="{cell.column_span}">'
                            f"{escape(cell.raw_text or '')}</td>"
                        )
                preview.append("</tr>")
            preview.append("</table>")
    preview.append("</html>")
    (directory / "table.html").write_text("\n".join(preview), encoding="utf-8")
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifact_sha256"].update(
        {name: sha256(directory / name) for name in ("evaluation.json", "table.html")}
    )
    manifest["evaluation_script_sha256"] = sha256(Path(__file__))
    write_json(manifest_path, manifest)
    return directory / "evaluation.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    print(evaluate(parser.parse_args().run))
