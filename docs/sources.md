# Official sources and reference snapshot

Reviewed on 2026-10-21. Sources below informed the architecture; proposed design
choices and untested behavior are distinguished in the architecture document.

CPU TATR assets and API availability were additionally checked on 2026-10-22;
see [CPU setup and model recommendations](cpu-setup.md). Both pinned TATR
checkpoints are downloaded to `models/tatr/` with a versioned SHA-256 catalog at
`src/jp_pdf_parser/tatr_checkpoints.json`. Their original model cards and processor
configs are preserved unchanged, and the MIT source license is included under
`docs/licenses/`. The reference-clone commands below still fetch source only.

## PaddleOCR

- [Official repository](https://github.com/PaddlePaddle/PaddleOCR)
- [PaddleOCR-VL usage tutorial](https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html)
- [Pinned Python pipeline wrapper](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/paddleocr/_pipelines/paddleocr_vl.py)
- [Pinned English usage guide](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)
- [Pinned PaddleOCR-VL 1.6 introduction](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md)
- [Pinned example pipeline configuration](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/deploy/paddleocr_vl_docker/pipeline_config_vllm.yaml)
- [Pinned package dependencies](https://github.com/PaddlePaddle/PaddleOCR/blob/dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf/pyproject.toml)

Local checkout: `.reference/PaddleOCR/`.

Reference commit: `dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf`.
Commit date: 2026-09-16. Upstream source is Apache-2.0 licensed.

The reference is a shallow, sparse checkout containing the relevant documentation,
wrapper, dependency metadata, and NVIDIA deployment examples. It is ignored by
the application repository. No upstream source was modified or executed.

The usage guide describes the full layout-plus-VLM pipeline, direct PDF/image
input, JSON export, page restructuring, and backend choices. The wrapper supports
explicit `v1.6` selection. The example configuration names `PP-DocLayoutV3` and
`PaddleOCR-VL-1.6-0.9B`; it is a source reference, not our verified runtime lock.
Financial reconciliation and the application schema are our own layers.

The revised design uses the documented `use_layout_detection=False` /
`prompt_label` path for externally segmented crops. There is no assumption that
these parameters accept an
arbitrary coordinate JSON or guarantee alignment from painted labels. The
[official PaddleX implementation](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/pipelines/paddleocr_vl/pipeline.py)
shows OCR and table task handling; that branch is a research reference, not a
runtime lock.

To recreate the reference after cloning this project, use PowerShell from the
repository root:

```powershell
git clone --no-checkout --filter=blob:none https://github.com/PaddlePaddle/PaddleOCR.git .reference/PaddleOCR
git -C .reference/PaddleOCR sparse-checkout set --no-cone /README.md /README_en.md /LICENSE /pyproject.toml /paddleocr/_version.py /paddleocr/_pipelines/paddleocr_vl.py /docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md /docs/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.en.md /deploy/paddleocr_vl_docker/accelerators/nvidia-gpu/ /deploy/paddleocr_vl_docker/pipeline_config_vllm.yaml
git -C .reference/PaddleOCR checkout --detach dab3fe35379033fdcb2d0e9572fac0b36c9a9ebf
```

The recreation commands use partial object fetching and a fixed commit. They do
not download model weights, install packages, or start inference services.

## Microsoft Table Transformer

- [Official Microsoft repository](https://github.com/microsoft/table-transformer)
- [Pinned inference guide](https://github.com/microsoft/table-transformer/blob/16d124f616109746b7785f03085100f1f6247575/docs/INFERENCE.md)
- [Pinned inference and cell reconstruction](https://github.com/microsoft/table-transformer/blob/16d124f616109746b7785f03085100f1f6247575/src/inference.py)
- [Pinned geometry postprocessing](https://github.com/microsoft/table-transformer/blob/16d124f616109746b7785f03085100f1f6247575/src/postprocess.py)
- [Detection checkpoint](https://huggingface.co/microsoft/table-transformer-detection)
- [Structure checkpoint: v1.1 All](https://huggingface.co/microsoft/table-transformer-structure-recognition-v1.1-all)
- [Transformers Table Transformer API](https://huggingface.co/docs/transformers/model_doc/table-transformer)

Local checkout: `.reference/table-transformer/`.

Reference commit: `16d124f616109746b7785f03085100f1f6247575`.
Commit date: 2023-09-06. Upstream source is MIT licensed. GitHub reports that the
repository was archived on 2026-09-21. Use the Transformers integration as the
proposed adapter; do not install the reference's historical environment wholesale.

The guide specifies PIL images and separate text/token inputs. Its cell-list
representation retains bounding boxes. The source detects structural objects
and reconstructs cells using row/column intersections and spans, with additional
token-assisted refinements. Our first pass will reconstruct geometry without
requiring earlier OCR, and preserve unresolved candidates. Task accuracy on
Japanese invoices and the overlay input proposal remain untested.

Recreate the source reference from the repository root:

```powershell
git clone --no-checkout --filter=blob:none https://github.com/microsoft/table-transformer.git .reference/table-transformer
git -C .reference/table-transformer sparse-checkout set --no-cone /README.md /LICENSE /environment.yml /docs/INFERENCE.md /src/inference.py /src/postprocess.py
git -C .reference/table-transformer checkout --detach 16d124f616109746b7785f03085100f1f6247575
```

The clone is shallow/sparse locally and ignored by the application repository.
No source was modified or executed and no model weights have been downloaded.
Checkpoint revision IDs and image processor settings will be pinned during
implementation. The Microsoft-hosted Hugging Face cards identify their author as
the Hugging Face team; primary implementation details were checked against the
Microsoft source.

## Japanese language model

Selected initial language validator, placed as the final model stage before JSON
export:
[LLM-jp 3 1.8B Instruct official model card](https://huggingface.co/llm-jp/llm-jp-3-1.8b-instruct).

The card identifies the National Institute of Informatics development team,
Japanese instruction-tuning data, Apache-2.0 licensing, Transformers usage, and a
4,096-token context window. It does not establish invoice OCR validation quality.
The weights/tokenizer revision is pinned in `config.toml`; runtime dependencies
are locked in `uv.lock`. No Japanese model weights have been downloaded yet.
PaddleOCR-VL fine-tuning is deferred under the current dataset and time constraints.

## PDF rasterization

[pypdfium2 official Python API](https://pypdfium2.readthedocs.io/en/stable/python_api.html)
documents PDF loading, rendering, resource lifetime, and PDFium's threading
constraints. The architecture proposes sequential rendering. The 300 DPI starting
point is our evaluation setting, not an official PaddleOCR requirement.

## Verification completed at this checkpoint

- Inspected the initially empty application workspace and Git configuration.
- Read official online documentation and cloned the official PaddleOCR source.
- Inspected the pipeline wrapper, dependency declarations, and model configuration.
- Recorded the exact reference commit and kept the checkout outside tracked code.
- Cloned the official Microsoft TATR source and inspected image inputs, structural
  classes, row/column refinement, cell reconstruction, and text-token dependence.
- Updated the diagram, module boundaries, geometry contracts, recognition input
  variants, and alignment evaluation plan.

The initial source review above was followed by the implementation described in
[progress.md](progress.md). Core code and PDF rendering now have automated tests;
model inference and Japanese invoice accuracy remain separate verification tasks.

## Local model pins and hosted model selection

The official Hugging Face model metadata was read to record these revisions:

| Component | Model | Revision |
| --- | --- | --- |
| TATR detection | `microsoft/table-transformer-detection` | `2357cbe2b5a5d1c03e54f32764f06058933b65ab` |
| TATR structure | `microsoft/table-transformer-structure-recognition-v1.1-all` | `7587a7ef111d9dcbf8ac695f1376ab7014340a0c` |
| Optional local PaddleOCR-VL | `PaddlePaddle/PaddleOCR-VL-1.6` | `c5630abae1d940eafe0697512a0325494b02ab42` |
| Optional local Japanese reviewer | `llm-jp/llm-jp-3-1.8b-instruct` | `413b7929df092690b0f9449f3045baaf8deba0a5` |

The default hosted model IDs are now `paddlepaddle/paddleocr-vl-1.6` on VLM Run
and `qwen/qwen3.8-27b` on Groq. Their returned IDs and usage are retained in raw
responses; the provider weight revisions cannot be pinned as local HF commits.
Both live catalogs and synthetic inference were verified with the user's keys.
Sources: [VLM Run methods](https://docs.vlm.run/gateway/methods),
[Paddle model contract](https://vlm.run/gateway/models/paddlepaddle-paddleocr-vl-1.6),
[Groq Qwen model](https://console.groq.com/docs/model/qwen/qwen3.8-27b),
[strict JSON output](https://console.groq.com/docs/structured-outputs).

The official PaddleOCR package release selected for implementation is `3.7.0`.
Its crop recognition path is exposed by the pinned wrapper, and the
[official PaddleX predictor](https://github.com/PaddlePaddle/PaddleX/blob/release/3.7/paddlex/inference/models/doc_vlm/predictor.py)
shows loading Hugging Face weights into the native Paddle model. This source
review does not substitute for a completed real-model inference test.
