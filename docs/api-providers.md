# Hosted API options

Checked 2026-10-22 against provider documentation and live public catalogs.
The selected VLM Run and Groq adapters are now implemented and enabled in the
default config. Provider research below records the reasons for that choice.
Live verification and remaining accuracy work are in [progress.md](progress.md).

## PaddleOCR-VL 1.6

Recommend **VLM Run Gateway** for the next integration experiment. Its
[exact model card](https://vlm.run/gateway/models/paddlepaddle-paddleocr-vl-1.6)
lists `paddlepaddle/paddleocr-vl-1.6`, image input, and explicit `ocr` / `table`
methods. Published rates are $0.15 per million input tokens and $0.35 per million
output tokens. Its [pricing page](https://vlm.run/pricing) advertises a $10 signup
balance without a card. This is trial credit, not permanently free inference.

Endpoint: `https://gateway.vlm.run/v1/openai/chat/completions`, using a VLM Run key.
Its [method contract](https://docs.vlm.run/gateway/methods) accepts a top-level
`method`: use `ocr` for clean cells and `table` for annotated table images. These
map naturally to I and H while Python retains the fixed grid and request IDs.

`table_format.py` now converts OTSL to HTML with checked rectangular spans for
the annotated comparison. Original responses are preserved. Painted row labels
were observed in a live annotated response and do not overwrite clean values.
The live catalog advertises no supported generation parameters for this model,
so the client omits `temperature` / `max_tokens` rather than sending unsupported
fields. Crop resizing and its inverse transform are recorded. Oversized clean
OCR text is rejected with evidence retained, without truncating or accepting it.

**OpenParser** is another exact-version host. Its
[service page](https://www.openparser.dev/paddleocr-api) publishes $1 per 1,000
parsed pages and $10 signup credit. Select `ocr_model="paddleocr-vl-1.6"` on
`POST https://api.openparser.dev/parse`. Each crop may be billed as a separate
image/page: our crop pipeline is not one API request per original PDF page.
Verify task controls for externally segmented cells against its
[model catalog](https://docs.openparser.dev/clients/models) before integration.
We do not need its separate schema-extraction LLM for our application-owned JSON.

**Official PaddleOCR API** also supports the exact version through its
[Python SDK](https://www.paddleocr.ai/latest/en/version3.x/inference_deployment/serving/paddleocr_official_api/python.html).
It needs a Baidu AI Studio token; crop/task mapping and account quota still need
verification. No permanent free allowance for our exact pipeline was established.

**OpenRouter and Groq:** the [live OpenRouter catalog](https://openrouter.ai/api/v1/models)
returned no PaddleOCR entries, and Groq's
[supported models](https://console.groq.com/docs/models) do not list it either.
Neither is currently a verified route for the requested model.

## Japanese review and free access

For free API prototyping, recommend Groq's
[Qwen3.8-27B](https://console.groq.com/docs/model/qwen/qwen3.8-27b), ID
`qwen/qwen3.8-27b`. This is a larger alternative to the previously suggested 4B
checkpoint; it runs remotely and does not require local model RAM or a GPU.
Japanese invoice review quality has not been measured here.

The [free-plan table](https://console.groq.com/docs/rate-limits) lists 30 requests
per minute, 1,000 per day, 8,000 tokens per minute and 200,000 per day. Account
limits may differ. Keep the account on the free plan; the developer plan has paid
rates. This is a preview model, so availability/allowances can change.

Groq's [API base URL](https://console.groq.com/docs/openai) is
`https://api.groq.com/openai/v1`. It requires a **Groq** key, not an HF token.
The model supports [JSON Schema output](https://console.groq.com/docs/structured-outputs),
but the application must still verify complete cell coverage and preserve OCR
amounts. Groq [retired the old Qwen3-32B on 2026-07-17](https://console.groq.com/docs/deprecations).

The live OpenRouter catalog returned no Qwen model with both input and output
prices equal to zero; old `qwen3-4b:free` links do not establish availability.
Hugging Face still has live providers for Qwen3-4B-Instruct-2507, but its current
[billing policy](https://huggingface.co/docs/inference-providers/pricing) gives
free users **no included inference credits**. Authentication does not establish
free inference. No paid model fallback has been enabled.

## Credentials, live use, and remaining work

The user supplied separate `VLMRUN_API_KEY` and `GROQ_API_KEY` entries in `.env`.
Both providers accepted their keys and listed the selected models. Keys are read
literally from the environment, or from `.env` with interpolation disabled.
The existing `HUGGING_FACE` entry is not used for these providers. Secrets and
authorization headers never enter configuration, errors, or run artifacts.

Run `jp-pdf-parser doctor` for dependency/key presence without network access;
`jp-pdf-parser check-api` explicitly makes small synthetic live requests and
writes its report and provider responses under `output/api-check/`. The default
`run` now uses CPU TATR plus both APIs. The `tatr` extra is sufficient; it does
not need Paddle or Japanese model weights locally.

Requests have explicit caps, spacing and timeouts, with no fallback models.
Network/auth/payment failures are not automatically retried. Groq's own
`json_validate_failed` response shares the existing one-format-repair budget;
the live-discovered case was reproduced and its repair verified. An OCR failure
can still consume the pipeline's one-clean-cell-per-table reread budget.

VLM Run responses include reported `usage.cost`; use those observations and the
provider billing ledger to budget crop-heavy processing, not just advertised
token-rate estimates. Stay on Groq's free plan if zero review charges are needed;
the application does not inspect or change the account billing plan. No signup,
subscription, or deployment was performed by the agent.

Still required: supplier/layout evaluation on representative Japanese invoices,
blank-cell hallucination handling, measured extraction accuracy, and review-rate
and cost/latency tuning. CPU TATR and the approved diagram's stage order and
fixed-grid ownership remain unchanged.
