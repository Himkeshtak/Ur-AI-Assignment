# Run the full architecture

`parse_document.py` is the entry point for the production architecture:

**PDF/image -> complete page -> CPU TATR detection -> CPU TATR structure -> fixed
grid and boxes -> full-page annotated image -> PaddleOCR-VL 1.6 -> deterministic
validation -> Japanese review -> JSON and PNG outputs.**

It calls `jp_pdf_parser.pipeline.Pipeline`. It does not import the no-TATR
experiment, crop tables/cells, or fall back to a different model.

From the repository root, run:

```powershell
.venv/Scripts/python.exe parse_document.py "balance_sheet_after_correction.pdf"
```

Replace the quoted path with a PDF or image. PNG, JPEG, BMP, single-frame TIFF
and WebP are supported. An active environment with the project's dependencies can
also run `python parse_document.py "input.pdf"`.

## One-time setup on a new machine

TATR runs on CPU. PaddleOCR-VL and Japanese review use the existing APIs.

```powershell
uv sync --locked --extra tatr
.venv/Scripts/python.exe -m jp_pdf_parser download-tatr
```

The downloader verifies the pinned TATR assets and skips valid files already
present in `models/tatr/`. Keep these keys in the repository's ignored `.env`:

```dotenv
VLMRUN_API_KEY=your_vlm_run_key
GROQ_API_KEY=your_groq_key
```

Default config, credentials and TATR checkpoint paths are resolved relative to the
configuration file, so invocation from another working directory also works. The
configured CPU TATR checkpoints are loaded locally without another backbone download.

## Outputs

The script prints a unique folder under `output/documents/` containing:

| File | Purpose |
| --- | --- |
| `simple.json` | Compact tables, original text, numbers, spans, real TATR boxes, axes and status |
| `simple.schema.json` | JSON Schema for the compact output |
| `result.json` | Complete production-pipeline result, including evidence references |
| `schema.json` | JSON Schema for the complete result |
| `pages/p001.png` | Complete clean source page, also saved if no table is found |
| `pages/p001-annotated.png` | Full page with table, row and column boxes and IDs, when tables are found |
| `tables/p1-t1/grid-g1.json` | Reconstructed fixed grid and coordinate ownership |
| `ocr/`, review, validation and manifest files | Original responses, issues, request counts and hashes |

PDFs produce one clean PNG per page and an annotated PNG for each page with detected
tables. Single-image inputs produce one clean PNG after EXIF orientation and white
background compositing. The API may resize a full-page copy to the model's pixel
budget; there is no cropping. All active boxes use rendered-page pixels.

The compact output includes `geometry_valid` for each table. Predicted boxes are
candidates: a structurally consistent grid still requires accuracy evaluation.
Mismatched OCR topology is retained as raw evidence, with cell text left unassigned.
The compact and full JSON files preserve the same status and issue references.

Image paths are relative to the output folder. Source text such as `2,000` stays
a string with its commas intact. Parsed decimal strings are separate fields;
amounts are never changed to make arithmetic balance.

## Options and limits

```powershell
.venv/Scripts/python.exe parse_document.py "C:/documents/invoice.pdf" --output-dir "C:/results" --financial-mode invoice
```

Generic-table mode is the default. Invoice mode uses `config.toml` financial
policies; it does not infer tax rates or rounding. Generic-table mode does not
perform balance-sheet reconciliation. `--config` selects a different config file.

Each page with detected tables uses one Paddle request; the document uses at most
one Groq request. Japanese review is skipped when no text could be assigned.
Automatic retries and review splitting are disabled. Token and account quotas
still apply. CPU startup/inference can take time on modest hardware.

Defaults: 30 PDF pages, 50 MB, 24 million raster pixels per page, and 500 table slots.
Multi-frame images are rejected; use PDF for multiple pages. Runs never overwrite
earlier output folders. Exit codes: `0` checks passed, `2` review needed, `1` failure.

## Improvements

See [the improvement plan](docs/improvements.md) for annotated Japanese-table data,
TATR and PaddleOCR-VL fine-tuning, LoRA feasibility, evaluation and deployment.
Training is proposed work; no dataset, trained adapter or edge benchmark is bundled.

The no-TATR control stays separate in [experiments/](experiments/README.md):

```powershell
.venv/Scripts/python.exe -m experiments.no_tatr "input.pdf"
```
