# Implementation checkpoint

The implementation follows the approved A-L pipeline with M between K and L.
This file distinguishes working code, tested behavior, and outstanding work.

## Code written

1. **Repository and configuration:** `pyproject.toml`, `uv.lock`,
   `.python-version`, and `config.toml` define the package, CLI, CPU defaults,
   limits, explicit financial policies, and immutable model revisions.
2. **A-B:** `pdf.py` inspects and hashes the PDF, enforces limits, renders pages
   sequentially, and records pixel-to-PDF transforms.
3. **C-D:** `structure.py` loads the pinned Microsoft detection and structure
   checkpoints through Transformers. Both official source references remain
   isolated under `.reference/`.
4. **E-G:** `grid.py` reconstructs rows/columns and merged cells, assigns IDs,
   handles rotated table crops, generates coordinate legends, and extracts clean
   crops with shared pixel boundaries.
5. **H-I:** `ocr.py` loads the pinned PaddleOCR-VL 1.6 checkpoint through the
   official PaddleOCR wrapper. Annotated tables and clean cells use distinct
   task requests and retain their original responses.
6. **J-K:** `tables.py` binds recognition to cell IDs and maps conservative
   invoice layouts. `validation.py` checks topology agreement, text encoding,
   numeric formatting, line amounts, subtotals, and totals using `Decimal`.
7. **M:** `japanese.py` runs LLM-jp 3 1.8B Instruct after deterministic validation.
   It budgets context, checks complete cell coverage, validates model JSON,
   bounds format retries, and stores suggestions as findings.
8. **L:** `pipeline.py`, `models.py`, and `artifacts.py` aggregate statuses and
   export typed JSON, validation reports, evidence, timings, dependency versions,
   and artifact hashes. `__main__.py` provides `run`, `doctor`, and `schema`.

## Verification

The model-free suite covers real PDF rendering and coordinate mapping, rotated
crops, merged/blank cells, separator corruption, full-width numbers, fractional
quantities, explicit rounding/tax policies, annotation contamination, partial
page failure, cell failure without row shifting, malformed reviewer JSON, and
review coverage. Injected model outputs are explicitly test fixtures, not real
recognition results. The fixture invoice is synthetic and authored in this repo.

Current local results: **57 tests passed** on Python 3.11.14. Ruff lint and format
checks passed. `schema` successfully exports the Pydantic contract without model
dependencies. The dependency lock includes the optional inference runtime.
The CI workflow is written and pinned to action commits; it has not run on GitHub
in this session. The wheel and source distribution both build successfully.

The complete optional runtime is installed in `.venv`. `uv pip check` reports
compatible installed dependencies. PyTorch 2.8.0 (CPU), Transformers 4.57.6 model
classes, Paddle 3.3.1, and PaddleOCR 3.7.0 / `PaddleOCRVL` import successfully in
one process. Paddle's standard user-cache initialization required execution
outside the workspace sandbox; the application model snapshots and PaddleX cache
are configured under `.cache/`. Import checks did not download model weights.
`doctor` reports all required dependencies present. Full checkpoint loading and
inference have not been verified.

## Still to do

- Run the full pipeline with the actual TATR, PaddleOCR-VL, and Japanese review
  weights on representative Japanese invoice PDFs. Core tests do not establish
  model accuracy or confirm every model-loading path.
- Measure row/column alignment, Japanese text error, exact amount match, review
  rate, memory use, and latency. Compare clean-cell and annotated-table results.
- Add supplier-specific header and financial profiles where samples justify
  them. Adjustment semantics and multiple tax bases currently require review.
- Verify GPU installation and memory settings on the target hardware. The default
  locked Paddle runtime is CPU-only.
- Add cross-page table association or structural re-detection retries if needed.
  The current automatic retry only re-reads failed/empty cell OCR, once per table.
- Fine-tuning and evaluation-dataset collection remain deferred as agreed.

There is no web application, database, training pipeline, or silent fake-model
mode. Model inference is invoked only by a real parse; tests inject their own
backends through the same pipeline interfaces.
