# Improving Japanese table extraction

The main command uses the full TATR + PaddleOCR-VL 1.6 architecture. Domain-specific
fine-tuning of both layout and recognition models is a promising next step. It
may improve Japanese table alignment, transcription and amount extraction, but a
small amount of fine-tuning does not guarantee better accuracy. Treat each change
as a measured experiment with held-out data.

## Build a well-annotated Japanese tabular dataset

Collect representative invoices, balance sheets and statements from the target
domain, using documents we are permitted to train on. Include borderless tables,
merged headers, blank cells, repeated labels, multi-line Japanese, negative amounts,
full-width numbers, thousands separators, dates, scans, skew and low-resolution
inputs. Correct the labels manually rather than trusting OCR output as ground truth.

For every page, annotate table boxes, row/column boxes, merged-cell ownership,
headers, exact cell text and reading order. Distinguish surrounding titles/units
from table contents. Retain original amount strings, their currency/unit context
and separately checked numeric interpretations. Financial totals alone do not
establish transcription accuracy.

Split training, validation and test data by supplier/template or document origin,
not by near-duplicate pages. Keep multi-page documents together. Version source
hashes, labels, preprocessing and adjudicated corrections. This dataset supports
both TATR layout supervision and Paddle image-to-table/text supervision.

## Fine-tune TATR detection and structure

Adapt both checkpoints where the measured errors justify it: detection learns
where Japanese tables start/end, while structure recognition learns rows, columns,
headers and spanning cells. Microsoft's repository provides ordinary checkpoint
fine-tuning and evaluation, including GriTS. [Official TATR training guide](https://github.com/microsoft/table-transformer#fine-tuning-and-other-model-training-scenarios).

Evaluate **LoRA on TATR** as a parameter-efficient option alongside head-only and
full fine-tuning baselines. This repository currently has no TATR LoRA integration.
The upstream training instructions do not supply a ready-made LoRA switch. Inspect
the actual attention/linear modules, configure supported adapter targets, and decide
which classification and box-regression heads remain trainable. Verify gradients,
trainable parameter counts, loss improvement and save/reload/merge equivalence before
scaling the training job. [PEFT custom-model guidance](https://huggingface.co/docs/peft/en/developer_guides/custom_models).

Training inputs must resemble deployment inputs. The present no-crop architecture
feeds complete pages to structure recognition, while Microsoft's reference extraction
path uses detected table regions. Annotate and evaluate the intended full-page setting
explicitly; fine-tuning cannot substitute for checking this distribution difference.
Keep span-conflict handling and grid assembly under evaluation independently.

## Fine-tune PaddleOCR-VL 1.6

PaddleOCR-VL 1.6 has a compact 0.9B-class architecture. Train on Japanese page/table
images paired with verified transcriptions and table markup, emphasizing row
association, metadata exclusion, Japanese characters and financial punctuation.
[PaddleOCR-VL 1.6 model documentation](https://www.paddleocr.ai/main/en/version3.x/algorithm/PaddleOCR-VL/PaddleOCR-VL-1.6.html).

Evaluate **LoRA or QLoRA** before committing to full-model training. Axolotl documents
an exact 1.6 multimodal fine-tuning path with QLoRA and full-fine-tuning examples;
its QLoRA example adapts the decoder, vision encoder and projector. Pin a compatible
training stack in a separate environment rather than replacing the inference lock.
Validate the input task/template and table targets, check which modules are frozen,
and measure Japanese accuracy after merging/export. [Axolotl's PaddleOCR-VL guide](https://docs.axolotl.ai/docs/models/paddleocr-vl.html).

The upstream Paddle documentation also describes supervised VLM fine-tuning via
ERNIEKit. Its generic 0.9B tutorial is not proof that every setting transfers to our
exact 1.6 checkpoint. Verify checkpoint compatibility before training.
[Official SFT guide](https://github.com/PaddlePaddle/ERNIE/blob/release/v1.4/docs/paddleocr_vl_sft.md).

LoRA reduces trainable parameters and optimizer-state costs; the base model and
activations still require memory. Adapter training does not itself shrink the
inference model or make a CPU/edge deployment fast. No LoRA training has been run
in this repository. [PEFT LoRA documentation](https://huggingface.co/docs/peft/en/developer_guides/lora).

Run pilot training in a separate compatible training environment, normally with
GPU acceleration. The local CPU inference check does not establish practical CPU
training. Determine the training memory budget from image sizes, sequence lengths
and trainable modules before scaling a run.

## Server and edge deployment

Both models are candidates for comparatively modest dedicated deployment. Our two
TATR weight files total about 231 MB on disk and CPU inference has been executed
locally. Paddle's 0.9B-class size also makes self-hosting worth evaluating. These
facts do not specify runtime RAM, concurrency, training memory or latency.

- **CPU server:** benchmark both TATR models, rasterization and OCR separately at
  the intended resolutions. Compare CPU-only Paddle serving with a modest supported
  accelerator where available. Training hardware requirements differ from inference.
- **Edge TATR:** consider a CPU/NPU/GPU-capable edge device with sufficient memory.
  Test model export, operator support and quantization against the intended runtime.
  Measure both detection and structure passes, peak RAM, latency and power on the
  actual device. Edge suitability is a hypothesis, not a validated repository feature.
- **Custom Paddle serving:** the current VLM Run model ID serves the provider's
  checkpoint. A locally trained LoRA adapter is not uploaded or activated by an API
  key. Add a verified custom-serving backend/provider route and merged-model loading
  before expecting fine-tuned weights to affect results.

Keep official downloaded weights immutable. Store fine-tuned models/adapters under
new versioned identities, with base revisions, dataset versions and checksums. The
current TATR checksum catalog must be extended deliberately for custom checkpoints;
overwriting the pinned official weights will correctly fail verification.

## Evaluate and promote in stages

First resolve/document current row and merged-cell failures and preserve a frozen
baseline. Compare unmodified models, each model fine-tuned separately, and both
fine-tuned together. Separate the effect of annotations from model training using
controlled image inputs; retain the no-TATR experiment as a control.

Measure table detection, GriTS/row-column ownership, merged-cell accuracy, Japanese
character error, exact amount/separator matches, missing/extra cells, review coverage,
review rate, API tokens/cost, peak memory and end-to-end latency. Report results on
unseen layouts; do not promise an accuracy percentage before testing. Retain exact
arithmetic rules and review gates even when language/OCR scores improve.

Other independent work remains: Japanese-review coverage, metadata contamination,
number-format handling with immutable raw text, and explicit balance-sheet rules.
Fine-tuning TATR/Paddle does not repair a reviewer that omits cell IDs, a provider
quota, an unsafe assignment rule or an unspecified financial policy.

No dataset collection, adapter training, accuracy gain, custom model hosting or edge
benchmark has been completed as part of this documentation update.
