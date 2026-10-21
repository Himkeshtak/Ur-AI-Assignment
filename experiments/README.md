# No-TATR experiment

This directory is independent of the production pipeline. No production scripts,
configuration, model assets or architecture files are changed by the experiment.

`no_tatr.py` renders the complete PDF page with the existing renderer, sends that
clean page directly to the existing PaddleOCR-VL 1.6 API adapter, and reconstructs
logical rows, columns and spans from its OTSL/HTML response. It makes no TATR call,
draws no annotations and creates no crops. Missing cell coordinates and layout
confidence are explicitly null; no uniform or fabricated boxes are supplied.

The existing financial rules, Japanese reviewer, coverage validation, status
aggregation and artifact writer are reused. Physical geometry validation is not
applicable without TATR; the experiment explicitly validates logical grid coverage
and encoding instead. Balance sheets use generic-table mode, which does not
implement balance-sheet arithmetic. Results remain `needs_review` because there
is no independent pixel-alignment check. Experimental schema changes stay here.

Run from the repository root with the existing environment and `.env` credentials:

```powershell
.venv/Scripts/python.exe -m experiments.no_tatr balance_sheet_after_correction.pdf --baseline output/balance-sheet-full-page/run-20261021T205235Z-a2d7b3bad5a2
.venv/Scripts/python.exe -m experiments.evaluate_no_tatr output/experiments/no-tatr/<run>
.venv/Scripts/python.exe -m pytest -q experiments/test_no_tatr.py
```

The first command uses one OCR request per page and at most one Groq request for
the document, with the existing model, resolution and review settings. There are
no automatic retries. The evaluation command is local only. To test an invoice,
select `--financial-mode invoice` and supply an appropriate financial profile.

Outputs include `result.json`, `cells.csv`, schema, validation, Japanese review,
raw provider responses, request counts, comparison and manifest. The evaluator
adds a safe escaped `table.html` preview and `evaluation.json`. Production-file
hashes are checked before/after execution and retained. Baselines must use the
same source hash, rendering settings, OCR model and pixel budget.

## Observed balance-sheet result

Run: `output/experiments/no-tatr/run-20261021T211114Z-ef3eb3ef4ea5/`.

| Measure | Existing TATR path | Clean-page experiment |
| --- | --- | --- |
| TATR grid | 44 rows, 3 columns, conflicting spans | Not used |
| Paddle output | 48 rows, 3 columns | 45 rows, 3 columns |
| Cells assigned an OCR reading | 0 | 134 (including blank cells) |
| New API calls in experiment | — | 1 Paddle, 1 Groq |

Paddle placed a correction note and the unit label in an extra first table row.
The clean response still does not match the saved TATR topology; its saved TATR
geometry also remains invalid. Groq returned only 64 of 134 required review IDs;
the unchanged coverage validator rejected that response. No suggestions were applied.

Eight selected numeric values from four visually checked source rows match after
ignoring whitespace for evaluation only. Raw numeric strings contain spaces after
commas and are retained unchanged. This is a limited spot check, not full accuracy
measurement or financial reconciliation.

The current TATR/grid reconstruction path is an observed assembly blocker on this
document. This does not isolate Microsoft's model as the sole cause: the experiment
also removes synthetic annotations and fixed-grid rejection, Paddle still includes
metadata as a table row, and Japanese review has an independent coverage failure.
Evaluate more annotated documents and separate those factors before replacing the
production path.
