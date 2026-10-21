# Local CPU TATR and hosted recognition

Updated 2026-10-22. The target machine has no GPU. Keep stages C-D on its CPU;
prepare image crops and validate/export JSON locally as before. Hosted PaddleOCR
and Japanese-review integration remains outstanding. The approved diagram and
cell-ID ownership do not change.

## Downloaded assets

| Directory | Official checkpoint | Pinned revision | Weight bytes |
| --- | --- | --- | ---: |
| `models/tatr/detection/` | `microsoft/table-transformer-detection` | `2357cbe2b5a5d1c03e54f32764f06058933b65ab` | 115,317,516 |
| `models/tatr/structure/` | `microsoft/table-transformer-structure-recognition-v1.1-all` | `7587a7ef111d9dcbf8ac695f1376ab7014340a0c` | 115,437,156 |

Each folder contains `model.safetensors`, `config.json`,
`preprocessor_config.json`, and the upstream `README.md`. Total locked assets:
230,835,538 bytes (about 231 MB / 220 MiB). The source reference remains in
`.reference/table-transformer/`; Microsoft's MIT license is preserved in
[licenses/table-transformer.txt](licenses/table-transformer.txt).

These are the published Microsoft checkpoints in Transformers format. Both need
PyTorch, torchvision, Transformers, timm, Pillow, and their locked dependencies.
There is no separate tokenizer, LoRA adapter, or additional ResNet download for
TATR. Its backbone weights are included in the checkpoints; the adapter sets
`use_pretrained_backbone=False` when constructing the model.

`src/jp_pdf_parser/tatr_checkpoints.json` records the immutable model revisions,
file sizes, and SHA-256 digests. On initial download, weights were compared with
Hugging Face's LFS SHA-256 metadata and small files with the pinned Git blob IDs.
The application verifies all locked files before local loading. The default
config selects these folders and `cpu` for both checkpoints.

Model binaries and Hugging Face transfer metadata are ignored by Git. Copy
`models/tatr/` when transferring the prepared repository to another CPU machine,
or rerun the downloader. The small checksum catalog is versioned with the code
and included in built packages.

## Reproduce on Windows

Run from the repository root with uv installed:

```powershell
$env:UV_CACHE_DIR = Join-Path $PWD '.cache/uv'
$env:UV_PYTHON_INSTALL_DIR = Join-Path $PWD '.cache/python'
uv sync --locked --extra tatr
uv run --locked --extra tatr jp-pdf-parser download-tatr

# Enforce offline operation for the verification run.
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
uv run --locked --extra tatr jp-pdf-parser check-tatr --output-dir output/tatr-check
```

`download-tatr` requires network access only for missing or damaged assets; it
uses public downloads without a token. Restore network access for future missing
downloads by removing the two offline environment variables. Local inference
uses `local_files_only=True` even without those variables.

The check saves input images, model/processor versions, file checksums, effective
resize settings, timings, and raw predictions. `--image data/page.png` uses a
supplied page image and the highest-scoring detected table crop for structure.
With no image, it generates a small synthetic English table. It tests model
execution, not Japanese recognition or alignment accuracy.

The structure processor's published config uses `size={"longest_edge":800}`.
Transformers 4.57.6's slow DETR processor requires both edge keys. The adapter
converts this in memory to equal shortest/longest limits of 800, preserving
aspect ratio and the longest-edge target. Upstream files remain byte-for-byte
unchanged. Detection uses the published 800/800 resize. The detection loader also
reports three unused `num_batches_tracked` buffers; these are training counters
for BatchNorm layers replaced by frozen normalization during inference.

## Japanese model recommendation, pending selection

Recommend [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
for API-based review. It is a 4B-parameter instruction model, supports only
non-thinking generation, and its model page currently lists Nscale as an
Inference Provider. The [Qwen3 language list](https://qwenlm.github.io/blog/qwen3/)
includes Japanese. It is multilingual, not specifically trained on this project's
invoices; suitability must be measured on Japanese OCR errors.

A smaller option is [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B),
currently listed with Featherless AI on the model page. Its non-thinking mode
reduces generation overhead when supported by the selected provider. Prefer
evaluating the 4B model first; this is an engineering recommendation, not a
measured quality comparison on our documents.

Hugging Face's [Nscale](https://huggingface.co/docs/inference-providers/providers/nscale)
and [Featherless](https://huggingface.co/docs/inference-providers/providers/featherless-ai)
integrations accept a Hugging Face token through its inference router, subject to
account permissions, access, and billing. No API call or deployment was made.
Current configuration and diagram still name LLM-jp 3 1.8B Instruct: Qwen is a
recommendation only, and no new language-model weights were downloaded.

The review model should return cell-referenced language findings. It cannot
verify glyphs from text alone, rewrite OCR amounts silently, or replace exact
`Decimal` financial checks.

## PaddleOCR-VL 1.6 endpoint requirement

The [exact model page](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6)
currently lists no Inference Provider. An HF token grants authentication but
does not create a running model service. HF-based integration also needs an
existing dedicated endpoint URL or a chosen Space/API with a verified request
contract. The project has not deployed one.

The [official PaddleOCR API](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/paddleocr_official_api/python.html)
does list version 1.6 but requires its own AI Studio token. That is a separate
service from Hugging Face. Keep the exact selected version and both annotated
table / clean-cell request branches when implementing any remote adapter.
