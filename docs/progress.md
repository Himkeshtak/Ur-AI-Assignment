# Full-architecture entry point

`parse_document.py` now calls the production `Pipeline`: local CPU TATR detection
and structure, fixed-grid/page annotations, PaddleOCR-VL, deterministic validation,
Japanese review and export. It no longer imports experiment code. The no-TATR
control remains in `experiments/`. Shared PDF/image intake supports both entry points
without cropping. See [usage](../PARSE_DOCUMENT.md) and [improvements](improvements.md).

Verification: 123 tests pass, including two TATR passes per input page, real box
coordinates in compact JSON, full-page image preservation and provider failures.
The real CPU run is `output/documents/run-20261022T045422Z-bedbd9cb56f2/`.
Its manifest identifies `TATR`, `VLMRunOCR`, and `GroqReviewer`, and all 15 artifact
hashes were checked. It saved compact/full JSON, their schemas and clean/annotated
PNGs. One OCR request was made. Japanese review was skipped because conflicting
spans prevented text assignment; status is correctly `needs_review`.

The fine-tuning/LoRA and server/edge section is a proposed improvement plan.
No training, accuracy improvement or edge deployment has been performed.

---

# Current pipeline update: full pages only

The current architecture is [architecture.md](architecture.md). It supersedes the
cell-crop recognition flow described in the historical implementation notes below.
The active code creates no table/cell crops: both TATR passes see a complete page,
one annotated full-page PaddleOCR request is sent per page, and Groq receives one
compact document-level review. The shipped config disables automatic retries and
batch splitting. Raw evidence is retained when topology or review fails.

The balance-sheet results from the earlier crop pipeline are historical outputs.
They do not validate the new recognition path. Live verification of the new path
and remaining alignment limitations are recorded separately below.

## Verification of the full-page revision

- **111 tests pass**, with Ruff lint and formatting checks passing. The tests forbid
  PIL cropping in the main page pipeline, check full-page dimensions, verify one
  OCR request per page even with multiple tables, and exercise a single compact
  Japanese-review request containing 240 cells.
- Live balance-sheet run: `output/balance-sheet-full-page/run-20261021T205235Z-a2d7b3bad5a2/`.
  It sent one PaddleOCR-VL request and one Groq request. The provider reported
  `$0.001` for the OCR response (not an independent billing-ledger verification).
  The complete 2481 x 3508 image was resized to 1065 x 1506 for OCR; no crop was made.
- That live run did **not** produce accepted cell text: TATR reconstructed 44 rows
  and conflicting merged spans; OCR returned a 48-row table. Assignment was rejected.
  Groq returned HTTP 429. A lower request count did not remove the token/quota limit.
- A guard now skips language review when no OCR text is mapped to cells. The final
  code was checked by a cached replay at
  `output/balance-sheet-full-page-replay/run-20261021T205714Z-5e9f957d67ce/`.
  The replay verified identical OCR input pixels, used no new network inference,
  checked all 14 artifact hashes and exported `needs_review` with the original
  raw table response intact. Its provenance identifies the live source run.
- **Remaining:** resolve full-page row/merged-cell alignment and evaluate Japanese
  review on successfully mapped documents. The full-page adaptation is not a
  production-accuracy result. Generic-table mode still has no balance-sheet
  arithmetic profile.

## Historical implementation notes

# Implementation checkpoint

The approved stage order is unchanged: PDF rendering, CPU TATR detection and
structure, fixed-grid/crop preparation, both PaddleOCR recognition branches,
deterministic checks, Japanese review, then JSON/report export.

## Changes completed

1. **Configuration and dependencies:** the default `config.toml` selects VLM Run
   PaddleOCR-VL-1.6 and Groq Qwen3.8-27B, alongside the pinned local CPU TATR assets.
   Typed backend variants keep hosted model IDs separate from local checkpoint
   revisions. httpx and python-dotenv are locked; reportlab is a development-only
   dependency for the synthetic fixture generator. `.env.example` contains no keys.
2. **API transport:** `api.py` loads literal credentials from environment or `.env`,
   enforces request caps/spacing/timeouts, rejects unexpected model IDs and
   incomplete completions, and sanitizes errors. Keys/headers are not exported.
   No automatic network retry, implicit provider choice, or fallback model exists.
3. **PaddleOCR:** `ocr.py` sends each clean image with `method="ocr"` and the
   numbered image with `method="table"`. Image resizing, hashes, inverse scales,
   returned model IDs, original responses and usage are retained. The live VLM Run
   catalog supports no generation parameters for this model, so unsupported
   max-token/temperature fields are omitted. Long clean-cell output is rejected
   with its original evidence retained, rather than accepted or silently truncated.
4. **Table adaptation:** `table_format.py` converts OTSL to HTML with explicit
   ownership and rectangular-span checks, preserving blanks and rejecting malformed
   continuations. This serves the annotated comparison only; it does not replace
   TATR geometry or clean-cell text. Unsupported topology becomes a review issue.
5. **Japanese review:** `japanese.py` shares coverage/chunking/format validation
   between explicit local and hosted backends. Groq uses strict JSON Schema and
   non-thinking generation. Every requested cell must be accounted for exactly
   once; suggestions are never applied to amounts or source text. A provider
   `json_validate_failed` rejection shares the single format-repair retry budget.
6. **Pipeline/export:** `pipeline.py` selects the configured adapters and preserves
   raw rejected OCR evidence without accepting it as a cell value. Config and
   response provenance record hosted aliases honestly, without invented weight
   commits. Earlier PDF limits, grid transforms, Decimal checks, and statuses remain.
7. **Tools/documentation:** `doctor` checks required dependencies and credential
   presence without network calls; `check-api` explicitly exercises both OCR tasks
   and Japanese review with synthetic input. `scripts/create_demo.py` creates a
   reproducible Japanese invoice using a supplied TrueType font. README, architecture,
   CPU/API setup, source references and CI commands reflect the current implementation.

The optional local PaddleOCR and LLM-jp adapters are retained as explicit choices,
not API fallbacks. Reference source clones and MIT attribution remain available.
No PaddleOCR or Japanese model weights were downloaded and no GPU is needed locally.

## Verification

**102 tests passed**, both in the full development environment and a separate core
installation without PyTorch/Paddle/model extras. Ruff lint/format checks passed.
Tests cover original PDF/grid/financial rules plus secret-safe errors, timeouts,
request limits/spacing, exact model selection, malformed/truncated responses,
OTSL spans/blanks, remote review coverage, oversized OCR evidence and provider-side
JSON-repair errors. Automated tests use mock HTTP transports, never real keys.

Both provided keys authenticated against their respective live model catalogs;
the exact configured models were listed. A live synthetic provider check made
**two VLM Run requests and one Groq request**. It preserved `2,000`, converted the
annotated OTSL response, and reviewed all four Japanese probe cells successfully.
Evidence: `output/api-check/run-20261021T200221Z-2e1e0ad2742f/`.

The generated Japanese invoice was rendered and visually inspected. The full
pipeline then ran with real local TATR and both real APIs. The final run extracted
**one table, six rows, four columns and 24 cells**; Groq reviewed every cell in two
requests. Amounts included observed `2,000` / parsed `2000`, and subtotal, 10%
exclusive tax and final total checks passed. The known demo tax profile was set
explicitly in that run and is captured in its manifest; default unknown tax policy
was not weakened.

Final evidence: `output/live-pipeline/run-20261021T201124Z-58a48946f7c2/`, including
`result.json`, `validation.json`, `manifest.json`, page/crop/overlay images, raw
provider responses and Japanese traces. Its 25 VLM Run responses reported a total
`usage.cost` of **0.029487**; this is provider-reported usage, not a billing-ledger
verification or a general per-invoice estimate.

The final result is correctly **needs_review**, not checks_passed. The annotated
reading includes painted labels/topology differences; clean OCR confused Japanese
and simplified Chinese glyphs and hallucinated punctuation in blank cells. Groq
flagged those observations without rewriting them. Line multiplication was not
verified on this demo because the unit-price header was corrupted by OCR, although
subtotal/tax/total reconciled. This exposes recognition limits rather than hiding
or auto-correcting them.

An earlier run exposed a 21K-character blank-cell hallucination and one Groq JSON
rejection. Output bounds and a bounded format-repair path were added with tests;
the previously rejected review chunk then succeeded within two live attempts.
The final full run completed review coverage. Synthetic checks establish execution
and error handling, not representative Japanese invoice accuracy.

TATR's 230,835,538 bytes of pinned assets remain verified under `models/tatr/`.
Its processors and both real checkpoints also passed offline CPU inference. The
structure processor compatibility conversion preserves upstream files. Wheel and
source-distribution builds and installed-dependency checks pass locally; GitHub CI
is configured but has not been executed remotely.

## Still to do

- Evaluate representative supplier/layout samples with ground-truth cell values;
  measure Japanese character error, exact amounts, alignment, and review rate.
- Address blank-cell hallucinations and corrupted header recognition with measured,
  evidence-preserving preprocessing/re-read policies. Existing bounds flag risk;
  they cannot prove a cell is blank or establish the correct Japanese glyph.
- Assess whether the required annotated-image pass improves results; it currently
  can recognize synthetic labels, so its readings stay secondary.
- Measure total request count, reported/provider-ledger costs, quotas and latency
  for realistic workloads. Use Groq's free plan if zero review charges are needed;
  the application does not inspect or change account billing plans.
- Add supplier-specific aliases and policies, multiple-tax-base/adjustment support,
  cross-page association or structural retries only when samples justify them.
- Fine-tuning/dataset collection remain deferred. No website, database, background
  service, cloud deployment, account signup or subscription was created.
