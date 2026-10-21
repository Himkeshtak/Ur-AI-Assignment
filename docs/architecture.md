# Architecture proposal

Review date: 2026-10-21. Revised to put Microsoft Table Transformer before
PaddleOCR-VL. The main flow diagram is in [README.md](../README.md). The initial
parser is implemented; [progress.md](progress.md) records verification and
remaining work. Recognition accuracy still requires real-document evaluation.

The diagram follows the user's supplied A-L pipeline exactly, with only the
Japanese model block M inserted between K and L. Additional model branches are
outside the current pipeline scope.

## Scope and implementation choices

Build one Python package with a command-line entry point. Accept one PDF per run,
process every page, extract simple tables, and apply invoice checks when the
document provides enough information to identify the financial fields.

Use two Microsoft Table Transformer (TATR) checkpoints:

- `microsoft/table-transformer-detection` locates tables in page images.
- `microsoft/table-transformer-structure-recognition-v1.1-all` locates structural
  elements within each table crop. Its training combines PubTables-1M and
  FinTabNet.c; Japanese invoice accuracy has not been established here.

Application postprocessing builds the cell grid from predicted rows, columns,
headers, and spanning cells. That grid supplies the row/column assignments.
PaddleOCR-VL 1.6 recognizes the text inside these externally defined regions.
Code associates each recognition request and response with its cell ID, rather
than asking the VLM to regenerate the final row assignments.

For clean cell crops, use the documented region-recognition path with
`use_layout_detection=False` and the OCR task prompt. This revises the original
full-page-only design: TATR now supplies table segmentation. The numbered table
image uses a separate annotated-table recognition request. Preserve original
model outputs and convert them into our schema in code; neither model directly
guarantees application-specific JSON.

Select `llm-jp/llm-jp-3-1.8b-instruct` as the initial Japanese language validator.
Its official model card describes Japanese instruction tuning. Run it after the
structural and deterministic financial checks, immediately before final status
assignment and JSON export. It has not been evaluated on this project's invoices,
so the model remains replaceable. PaddleOCR-VL fine-tuning is deferred for this
phase because a suitable dataset and training time are unavailable. No custom
training is included in the current scope.

Proposed libraries are `pypdfium2` for rasterization, Pillow for image handling,
Transformers/PyTorch for TATR and Japanese text review, PaddleOCR for text
recognition, and Pydantic for typed contracts. Arithmetic uses Python's
standard-library `Decimal`.
Use `argparse` for the initial CLI and TOML for configuration. Avoid a web server,
database, queue, or agent framework until a concrete requirement calls for one.

Python 3.11.14, the dependency graph, and model revisions are now pinned. Core
tests run in the repository's managed virtual environment; the workstation's
system Python is 3.13.5. Model-runtime verification is tracked separately in
progress.md. Direct local inference is the initial integration path;
GPU serving can use Linux/WSL2 or containers after hardware is established. Native
Windows support for every acceleration backend is not assumed.

## Pipeline responsibilities

| Stage                 | Responsibility                                                                                                                                                                         | Saved evidence                                                                 |
| --------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| Intake                | Check file type, readability, encryption, page count, and configured limits. Hash the original bytes. Reject an input that cannot be processed.                                        | Input digest, page count, intake issues.                                       |
| Rendering             | Rasterize pages sequentially into lossless RGB images. Close page resources promptly.                                                                                                  | Page image, dimensions, effective DPI, coordinate transform.                   |
| Table detection       | Run TATR detection on each page and extract padded, orientation-correct table crops.                                                                                                   | Raw detections, scores, table boxes, transforms.                               |
| Structure recognition | Run TATR structure recognition on each clean table crop.                                                                                                                               | Raw row, column, header, and spanning-cell objects.                            |
| Grid reconstruction   | Resolve structural candidates, build cells and spans, assign IDs, and check geometry.                                                                                                  | Versioned grid JSON, row/column/cell boxes, geometry issues.                   |
| Recognition inputs    | Extract clean cell crops and generate numbered table overlays.                                                                                                                         | Crop-to-cell request manifest, annotated images, annotation metadata.          |
| Recognition           | Read clean cells and run the requested annotated-table pass.                                                                                                                           | Untouched PaddleOCR results, request IDs, input variant.                       |
| Adaptation            | Attach text to the fixed grid; map known headers and totals without changing cell ownership.                                                                                           | Generic table cells, optional invoice view, competing readings.                |
| Schema validation     | Check structure, spans, coordinates, field types, references, cell coverage, and page coverage.                                                                                        | Machine-readable structural and alignment issues.                              |
| Financial validation  | Parse amounts strictly and evaluate applicable document formulas with exact decimals.                                                                                                  | Rule, operands, expected value, observed value, difference, source references. |
| Japanese review       | As the final model stage before export, use LLM-jp 3 1.8B Instruct to flag suspicious wording and uncertain field interpretation using table/header context and earlier rule findings. | Prompt/model revision, structured findings, generation status.                 |
| Resolution            | Combine required checks and assign a document status. A bounded targeted re-read returns through validation, including Japanese review, before export.                                 | Both recognition attempts, unresolved issues, retry count.                     |
| Export                | Write result JSON, a validation report, and a run manifest atomically.                                                                                                                 | Complete run folder and artifact digests.                                      |

PaddleOCR also accepts PDF input directly. We propose explicit page rendering to
control resolution, retain evidence images, and make retries reproducible. This
is our implementation choice, not a requirement imposed by PaddleOCR. Send the
rendered page to TATR first and derive all table/cell images from the same source.
Do not rasterize the same PDF again merely to call another model.

## Table structure and alignment

TATR accepts images; it does not directly transcribe Japanese text. Its detection
and structure models are separate. Ordinary cell boxes are reconstructed from
row/column intersections and spanning-cell predictions. A detection box alone
is not a ready-made, guaranteed-correct table grid.

1. Run page-level table detection, retain scores, and crop each table with recorded
   padding. Handle the rotated-table class with an invertible transform.
2. Use the selected checkpoint's image processor and recorded resize/padding
   settings. Map all predictions back to the original rendered page. A 300 DPI
   source does not mean the neural network consumes that resolution unchanged.
3. Apply configurable class thresholds and deterministic duplicate suppression.
   Sort rows top-to-bottom and columns left-to-right in the upright table frame.
   Preserve raw predictions alongside refined geometry.
4. Intersect rows and columns to create ordinary cells. Resolve spanning-cell
   predictions into contiguous row/column ranges. Represent a merged cell once;
   do not duplicate its content into every covered position. Preserve blank cells.
5. Check positive areas, containment, ordering, duplicate ownership, conflicting
   spans, and unsupported gaps/overlaps. Unresolved structure means
   `needs_review`; recognizing more text cannot repair uncertain geometry by
   itself. Do not promote malformed grids to invoice validation.
6. Assign one-based row/column indices and IDs such as `p1-t1-g1-r2-c4`. Include
   the grid revision (`g1`) because a structural retry may change segmentation.
   Rows and columns also receive IDs and bounding boxes. Bind OCR requests to
   this frozen grid revision; a returned text string cannot change its owner.

The initial geometric pass must work without OCR tokens. Microsoft's reference
postprocessing also supports token-assisted refinements, which must remain a
separate step if used later. Do not remove a row or cell solely because no text
was supplied, or report a token-assignment confidence score when there were no
tokens. Preserve uncertainty on blank-versus-unreadable cells.

Store boxes as `[x0, y0, x1, y1]`, with a top-left origin in rendered-page pixels
and half-open bounds. Also record the upright-table/crop transforms and inverse
mapping to the original PDF coordinate system. If a margin is added for labels,
record its offset separately. Geometric scores are not transcription confidence.

### Numbered overlays and PaddleOCR-VL handoff

Generate the image requested by the user: draw table, row, column, and cell boxes
and label them with IDs and coordinates. Use thin outlines and place label text
on borders or in an added margin with leader lines to avoid covering invoice
glyphs. Save a matching JSON map so coordinates do not have to be read from
pixels. Keep an untouched table image and clean cell crops.

Send the numbered table image to PaddleOCR-VL in a separate annotated-table
recognition pass. This is an evaluation input required by this design, not a
documented coordinate-conditioning feature of PaddleOCR. The public wrapper's
task prompt parameter is not a general bounding-box JSON input. The JSON map
remains with the orchestrator; painting a cell ID does not guarantee the VLM will
return that ID or obey it. Do not assume multi-image paired input is supported.

Use clean cell crops for the cell text stored in the final grid. Each request
carries an application-side cell ID; the response is attached by that request ID,
not by a guessed row position or an echoed model-generated ID. Keep bounded
micro-batches and per-crop errors; never shift later responses into failed slots.
Recognize each merged cell once. Include small, bounded padding that avoids
neighboring cell text; preserve multiline text within its cell. Empty model
output is not automatic proof that a cell was blank.

Retain the annotated-table response as a separate reading. Synthetic labels,
coordinate digits, and line markings must not become extracted invoice values.
Do not fill missing clean-cell results from this response by ordinal position.
Compare compatible readings, flag disagreements, and use clean source imagery
for targeted re-reads. All original outputs remain available for review.

This pipeline extracts content through the two table-recognition branches in the
diagram. If a required total is outside every detected table, record a missing
operand and flag incomplete reconciliation.

This approach isolates row/column assignment from VLM table reconstruction.
TATR can still miss or missegment rows, especially on layouts unlike its
training data. More model passes and cell crops may increase latency. Measure
alignment accuracy and throughput rather than assuming either improvement.

## Image policy

- Start experiments at **300 DPI**, using lossless PNG and a white background.
  This is a proposed baseline, not a documented universal model requirement.
- Keep aspect ratio and record any PDF rotation, image rotation, crop, or resize.
  Preserve a mapping from processed image coordinates back to the original page.
- Enforce configurable page, image-pixel, and memory limits before allocation.
  Record any resolution reduction. Flag pages that exceed supported quality or
  resource limits instead of silently omitting them.
- Let TATR define the table grid and let the application prepare clean cell crops.
  Apply each model's documented processor and pixel limits. Lock the effective
  parameters after compatibility testing; retain high-resolution source images.
- Keep document unwarping and aggressive denoising or binarization off initially.
  Enable orientation correction or other transformations only when evaluation
  justifies them, and record the resulting transforms.
- For a flagged table, optionally render its region again at higher effective
  resolution, within limits and with padding and header context. Preserve the
  first result. Stop after one targeted retry per table, shared between structure
  and text retries. For a geometry issue rerun structure recognition and create
  a new grid revision; for a text issue keep the grid and re-read affected cells.

PDFium calls will be sequential in the first implementation. Its documentation
warns against simultaneous PDFium calls across threads.

## Data contracts

Pydantic models will be the single source of truth; export JSON Schema from them
instead of maintaining a second handwritten schema.

| Contract     | Main fields                                                                                                                                                               |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Document     | Schema version, source hash, total pages, page results, optional invoice view, status, issues.                                                                            |
| Page         | One-based page number, original geometry, rendered geometry, transform, tables, processing status.                                                                        |
| Table grid   | Table ID, grid revision, page reference, TATR model revisions, raw/refined table region, rows, columns, cells, geometry status.                                           |
| Row / column | Revision-scoped ID, one-based index, original detection score, refined bbox, coordinate frame.                                                                            |
| Cell         | Revision-scoped ID, row/column positions and spans, derived bbox, parent row/column IDs, raw text, normalized text, optional parsed number/currency, evidence references. |
| OCR request  | Request ID, cell/grid ID or annotated-table ID, crop path/hash, transform, clean/annotated variant, completion state, original response.                                  |
| Invoice view | Explicit mappings from source cells or text blocks to quantities, prices, line amounts, subtotal, tax groups, adjustments, and total.                                     |
| Issue        | Code, severity, stage, field/cell references, observed value, expected value if applicable, explanation, evidence.                                                        |
| SLM review   | Completion state, reviewed cell IDs, and schema-validated findings with issue codes and optional suggestions.                                                             |
| Run manifest | Input/artifact hashes, application revision, effective config, dependencies, model revisions, prompts, device/backend, and timings.                                       |

Raw source strings are immutable. Normalized strings are separate fields. Use
Unicode normalization on a parsing copy of numeric text to handle full-width
digits and punctuation; do not overwrite company names or descriptions.

Serialize decimal values as JSON strings to avoid binary floating-point loss.
This is a subset of an actual cell record; `schema.json` defines the full contract:

```json
{
  "id": "p1-t1-g1-r2-c4",
  "row": 2,
  "column": 4,
  "row_span": 1,
  "column_span": 1,
  "raw_text": "2,000",
  "normalized_text": "2,000",
  "number": "2000",
  "bbox": [1240, 600, 1540, 650],
  "coordinate_frame": "rendered_page_pixels",
  "bbox_source": "tatr_grid_reconstruction"
}
```

Currency is not inferred from Japanese language; the configured financial profile
sets the supported monetary scale. Coordinates above are illustrative. Cell boxes
come from explicit TATR grid reconstruction, not an assumption that PaddleOCR
returns cell geometry. A cell without usable geometry cannot enter cell-crop
recognition; retain it as an unresolved structural candidate. Scores from either
layout model must not be presented as text transcription confidence.

Preserve generic tables even when invoice mapping is uncertain. An unresolved
mapping is reviewable data, not a reason to fabricate invoice fields. Associate
cross-page rows only when continuation evidence is explicit. Otherwise preserve
page boundaries and flag incomplete financial reconciliation.

## Numerical and financial rules

1. Validate thousands grouping before removing commas. `2,000` becomes
   `Decimal("2000")`; `20,00` is invalid grouping. Handle currency marks,
   full-width characters, and supported negative conventions explicitly.
2. Never globally interchange commas and decimal points. `2.000` may be a
   legitimate decimal in some fields; when it could represent a damaged `2,000`,
   or conflicts with a whole-yen amount policy, record the ambiguity. Fractional
   quantities and unit prices remain possible.
3. Check `quantity * unit_price`, adjustments, line amounts, subtotals, tax groups,
   and the final total only when operands and their meanings are established.
   Do not apply one formula to all invoice layouts.
4. Represent tax-inclusive/exclusive prices, multiple tax groups, discounts,
   shipping, and other adjustments explicitly when present. Obtain rates and
   rounding rules from document evidence or a named configuration profile.
   No fixed tax rate or unrecorded rounding assumption is built into extraction.
5. Apply configured rounding at the documented stage. Report both the rule and
   the difference; do not silently add a tolerance to make totals balance.
6. Missing operands or unknown financial policies produce an incomplete check,
   never a successful check. In invoice mode, missing required reconciliation
   prevents `checks_passed`. In an explicitly selected generic-table mode,
   inapplicable invoice checks are reported as such.

A matching total does not prove a transcription is correct: several wrong cells
can still balance. Never change a recognized value merely to satisfy arithmetic.

## Japanese language model boundary

The selected model is `llm-jp/llm-jp-3-1.8b-instruct`. This is the final model stage
before writing the final JSON file. The model receives draft structured data,
table headers, nearby text, and completed deterministic rule findings. It returns
cell-referenced review findings; Python attaches those findings, computes the
final status, validates the output contract, and serializes the result. No model
is asked to regenerate the complete final document.

It flags likely OCR artifacts, broken Japanese text, inconsistent
labels, or ambiguous column meanings. Domain-specific names can be legitimate,
so unusual wording alone is not grounds for silently changing source data.

A text-only model cannot compare characters with pixels. Its suggestions are
review findings. Only a further visual recognition attempt or a human can supply
new visual evidence; a human-approved correction is saved as a separate decision
with its provenance.

Require structured output and validate that all referenced cell IDs exist.
Constrain generation when the selected backend supports it; never assume that
the model will always obey JSON instructions. Invalid output, a timeout, or an
unavailable model means this required stage is incomplete. Request one bounded
format retry, then mark `needs_review` if still incomplete. Treat document text
as data, not as instructions to the reviewer model.

The model's documented context window is 4,096 tokens. Split long tables into
bounded groups, repeat the relevant headers and minimal surrounding context, and
record which cells were reviewed. Do not silently truncate a document. Start
with greedy decoding and record parameters; exact reproduction across different
devices or inference engines is not promised.

## Resolution and result status

| Status            | Meaning                                                                                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `checks_passed` | Every page was accounted for, required checks completed, and no unresolved issue blocks the configured task. This is a validation outcome, not a guarantee of perfect OCR. |
| `needs_review`  | Useful output exists, but ambiguity, mismatch, missing content, uncertain mapping, or an incomplete required check remains.                                                |
| `failed`        | Intake or a required processing stage could not produce usable output. Save the error and available diagnostics.                                                           |

Per-page and per-check states preserve partial failures. A failed page prevents a
document-wide `checks_passed`. A PDF with no detected tables requires review for
this task. Unresolved grid geometry, unrecognized cells, annotation contamination,
or disagreement between readings also prevents `checks_passed`. The overall
status aggregates every required check; the SLM cannot override a geometry or
numerical failure.

Any implementation-level text retry runs only while that table's shared retry
budget remains; the main diagram shows only the requested stage sequence. A
structural retry re-enters grid reconstruction with a new
revision and regenerates all affected OCR requests. Re-map and revalidate the
candidate result through the same stages.
Keep both attempts and any disagreements. A changed monetary reading remains
reviewable until the disagreement is resolved; arithmetic consistency alone
cannot select the winning reading. Review output is still exported when the
budget is exhausted. Human review is a JSON/evidence handoff in the first version;
a review interface is outside the initial repository scope.

## Proposed repository structure

This structure is now implemented. See [progress.md](progress.md) for the exact
implementation scope, verification evidence, and remaining work. Model inference
accuracy is separate from core pipeline verification.

```text
.
|-- README.md
|-- .gitignore
|-- .gitattributes                # Consistent source line endings
|-- .github/workflows/ci.yml      # Locked core checks on GitHub
|-- .python-version              # Verified Python patch version
|-- pyproject.toml               # Package, CLI, development checks
|-- uv.lock                      # Verified dependency lock
|-- config.toml                  # Models, thresholds, rendering, financial policy
|-- src/
|   `-- jp_pdf_parser/
|       |-- __init__.py
|       |-- __main__.py          # Thin argparse CLI
|       |-- config.py            # Typed TOML configuration
|       |-- models.py            # Contracts and JSON Schema export
|       |-- pipeline.py          # Stage order, retries, status aggregation
|       |-- pdf.py               # Intake, rendering, geometry
|       |-- structure.py         # TATR detection and structure model adapters
|       |-- grid.py              # Geometry, spans, transforms, IDs, coverage
|       |-- ocr.py               # Paddle cell and annotated-table recognition
|       |-- tables.py            # Bind cell readings and map invoice fields
|       |-- validation.py        # Structural, numeric, and financial rules
|       |-- japanese.py          # Small-model review and output validation
|       `-- artifacts.py         # Run manifest and atomic JSON writes
|-- tests/
|   |-- test_numbers.py          # Separators, full-width digits, signs
|   |-- test_financials.py       # Reconciliation and rounding policies
|   |-- test_grid.py             # Alignment, transforms, blank/merged cells
|   |-- test_tables.py           # Structure, merged cells, provenance
|   |-- test_pipeline.py         # Failures, incomplete checks, bounded retries
|   |-- test_pdf.py              # Real PDF rasterization and resource limits
|   |-- test_ocr.py              # Paddle adapter contract
|   |-- test_japanese.py         # Reviewer JSON and coverage
|   `-- fixtures/               # Small synthetic/licensed PDFs and expected data
|-- docs/
|   |-- architecture.md
|   |-- sources.md
|   `-- progress.md
|-- .reference/PaddleOCR/        # Ignored, pinned upstream reference checkout
|-- .reference/table-transformer/ # Ignored, pinned Microsoft reference
|-- data/                       # Ignored local input PDFs
`-- output/                     # Ignored generated run artifacts
```

All application logic and tests are Python. TOML, Markdown, JSON, and Mermaid
serve configuration, contracts, documentation, and diagrams. Upstream PaddleOCR
is a dependency/reference, not code copied into the application package.

An output folder will contain `result.json`, `validation.json`, `manifest.json`,
`schema.json`, page images, raw TATR predictions, versioned `grid.json` files,
numbered overlays, clean crops, `ocr_requests.json`, and original Paddle outputs
for every input variant. Include retry crops and attempt outputs when applicable.
A document hash identifies the input; each run receives a separate directory to
prevent evidence from being overwritten. Do not log full invoice content by
default.

## Reproducibility and acceptance

During implementation, commit the dependency lock, effective example config,
prompt templates, schema version, and tested model revision IDs. Never use a
mutable `main` model revision or `latest` container tag as the production lock.
Record GPU/CPU, engine, package versions, and renderer settings in each run.

Tests must cover separator corruption, full-width numerals, decimal quantities,
negative amounts, mixed tax groups, rounding differences, merged cells, page
failures, malformed upstream output, invalid SLM JSON, missing totals, and retry
exhaustion. Include rotated tables, crop padding/resize round trips, duplicate
rows, missing columns, merged headers, blank cells, multiline Japanese, conflicting
spans, annotation-digit contamination, missing or out-of-order OCR responses,
outside-table totals, and grid revision changes. Keep model-free unit tests
separate from real model integration checks; mocked OCR does not establish
recognition quality.

Evaluate real end-to-end output on annotated Japanese invoices before calling
the parser ready. Measure table/cell alignment, Japanese text error, exact amount
match, detected financial discrepancies, review rate, and latency. Split
evaluation by supplier/layout so near-duplicate invoices do not inflate results.
Compare the original Paddle-only baseline, TATR with clean cell recognition, and
TATR with the requested overlay pass. Measure whether cell texts land in the
correct rows and columns, with a structure metric such as GriTS where applicable.
Record added model latency and total crop count. Accuracy targets and final
model/input settings depend on those measurements.

flowchart TD
    A["PDF"] --> B["Render page images"]
    B --> C["TATR: detect tables"]
    C --> D["TATR: recognize structureRows, columns, headers, merged cells"]
    D --> E["Reconstruct cell gridAssign IDs and bounding coordinates"]
    E --> F["Numbered table imageBoxes and coordinate labels"]
    E --> G["Clean cell cropsEach tied to a cell ID"]
    F --> H["PaddleOCR-VLAnnotated-table recognition"]
    G --> I["PaddleOCR-VLCell text recognition"]
    E --> J["Assemble JSON against the fixed grid"]
    H --> J
    I --> J
    J --> K["Validate alignment, Japanese text,number formatting and arithmetic"]
    K --> M["Japanese language validationLLM-jp 3 1.8B Instruct"]
    M --> L["Result JSON and review report"]
