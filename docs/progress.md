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
   checkpoints through Transformers on CPU. `checkpoints.py` and the packaged
   `tatr_checkpoints.json` catalog download and verify all eight official assets.
   Both checkpoints are present in `models/tatr/` and verified offline. Source
   references remain isolated under `.reference/`.
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
   and artifact hashes. `__main__.py` provides `run`, `doctor`, `schema`,
   `download-tatr`, and `check-tatr`.

## Verification

The model-free suite covers real PDF rendering and coordinate mapping, rotated
crops, merged/blank cells, separator corruption, full-width numbers, fractional
quantities, explicit rounding/tax policies, annotation contamination, partial
page failure, cell failure without row shifting, malformed reviewer JSON, and
review coverage. Injected model outputs are explicitly test fixtures, not real
recognition results. The fixture invoice is synthetic and authored in this repo.

Current local results: **64 tests passed** on Python 3.11.14. Ruff lint and format
checks passed. `schema` successfully exports the Pydantic contract without model
dependencies. The dependency lock includes the optional inference runtime.
The CI workflow is written and pinned to action commits; it has not run on GitHub
in this session. The wheel and source distribution both build successfully.

The complete optional runtime is installed in `.venv`. `uv pip check` reports
compatible installed dependencies. PyTorch 2.8.0 (CPU), Transformers 4.57.6 model
classes, Paddle 3.3.1, and PaddleOCR 3.7.0 / `PaddleOCRVL` import successfully in
one process. Paddle's standard user-cache initialization required execution
outside the workspace sandbox; the application model snapshots and PaddleX cache
are configured under `.cache/`. `doctor` reports all required dependencies present.

On 2026-10-22, downloaded both actual TATR checkpoints plus their processor/config
files and model cards: 230,835,538 bytes. Initial downloads were checked against
upstream LFS SHA-256 or Git blob digests; every local file now has a locked SHA-256.
No PaddleOCR or Japanese model weights were downloaded. A separate `tatr` extra
allows installing the CPU table stages without the Paddle runtime.

Both real TATR models loaded and completed inference in a network-restricted
process with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. The synthetic check
found one table; its structure passed through the actual grid, overlay, and crop
utilities into four rows, four columns, and 16 cells with no geometry issues.
Evidence is under `output/tatr-check/`: `report.json`, images, raw predictions,
`grid.json`, `numbered.png`, and cell crops. Grid/overlay generation was an
additional manual integration check; `check-tatr` itself writes the model report
and two input images. This is execution evidence on synthetic English content,
not a Japanese alignment benchmark.

In the recorded run, inference took approximately 0.44 seconds for detection and
0.43 seconds for structure (12 PyTorch threads, default processors). Loading took
7.68 and 0.53 seconds, respectively; first-load timing includes library overhead.
These are single-run observations on this workstation, not throughput guarantees.
The structure checkpoint's longest-edge-only processor config needed an in-memory
compatibility conversion for Transformers 4.57.6; regression tests cover this and
checksum failure before model loading. Upstream files remain unchanged.

[CPU setup](cpu-setup.md) records local reproduction and hosted API choices.
Qwen3-4B-Instruct-2507 is recommended for hosted Japanese review but has not been
selected in config, integrated, or evaluated. No credentials were collected and
no remote inference or deployment was performed.

## Still to do

- Integrate the chosen PaddleOCR-VL 1.6 API. The exact Hugging Face model currently
  has no listed Inference Provider, so its token alone is insufficient: an actual
  endpoint URL/Space request contract is needed. The current `run` adapter remains
  local. Do not silently fall back to a different model/version.
- Select and integrate the Japanese review API, then evaluate its findings and
  structured response coverage. Configuration still selects local LLM-jp.
- Run the full pipeline with real Japanese invoices, the downloaded CPU TATR
  checkpoints, and both remote adapters. Core tests and synthetic inference do
  not establish Japanese invoice accuracy.
- Measure row/column alignment, Japanese text error, exact amount match, review
  rate, memory use, and latency. Compare clean-cell and annotated-table results.
- Add supplier-specific header and financial profiles where samples justify
  them. Adjustment semantics and multiple tax bases currently require review.
- Measure CPU memory, latency, and worker/thread settings on representative pages.
  GPU installation is outside the target-machine plan.
- Add cross-page table association or structural re-detection retries if needed.
  The current automatic retry only re-reads failed/empty cell OCR, once per table.
- Fine-tuning and evaluation-dataset collection remain deferred as agreed.

There is no web application, database, training pipeline, or silent fake-model
mode. Model inference is invoked only by a real parse; tests inject their own
backends through the same pipeline interfaces.
