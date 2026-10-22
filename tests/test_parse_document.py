import json
from pathlib import Path
from types import SimpleNamespace

from conftest import write_pdf
from PIL import Image

from jp_pdf_parser.artifacts import sha256
from jp_pdf_parser.japanese import ReviewRun
from jp_pdf_parser.models import Detection, JapaneseReview
from jp_pdf_parser.ocr import OCRPrediction
from parse_document import Result, load_config, run

HTML = "<table><tr><td>品目</td><td>金額</td></tr><tr><td>紙</td><td>2,000</td></tr></table>"


class Structure:
    def __init__(self):
        self.calls = []

    def detect(self, image):
        self.calls.append(("detect", image.size))
        w, h = image.size
        return [Detection(label="table", score=1, bbox=(w * 0.1, h * 0.1, w * 0.9, h * 0.9))]

    def recognize(self, image):
        self.calls.append(("recognize", image.size))
        w, h = image.size
        return [
            Detection(label="table row", score=1, bbox=(w * 0.1, h * a, w * 0.9, h * b))
            for a, b in ((0.1, 0.5), (0.5, 0.9))
        ] + [
            Detection(label="table column", score=1, bbox=(w * a, h * 0.1, w * b, h * 0.9))
            for a, b in ((0.1, 0.5), (0.5, 0.9))
        ]


class OCR:
    def __init__(self, error=False):
        self.api = SimpleNamespace(requests=0)
        self.sizes = []
        self.error = error

    def recognize(self, path, task):
        self.api.requests += 1
        with Image.open(path) as image:
            self.sizes.append(image.size)
        assert task == "table" and path.suffix == ".png"
        if self.error:
            raise RuntimeError("Deliberate provider failure")
        return OCRPrediction(text=HTML, raw={"text": HTML})


class Reviewer:
    def __init__(self):
        self.api = SimpleNamespace(requests=0)

    def review(self, document):
        self.api.requests += 1
        assert document.checks
        return ReviewRun(
            report=JapaneseReview(
                completed=True, reviewed_cell_ids=[c.id for c in document.all_cells()]
            )
        )


def test_public_script_exports_simple_json_schema_and_one_png_per_pdf_page(config, tmp_path):
    config.financial.mode = "table"
    source = write_pdf(tmp_path / "two-pages.pdf", pages=2)
    ocr, reviewer, structure = OCR(), Reviewer(), Structure()
    directory, result = run(
        source, config, tmp_path / "output", structure=structure, ocr=ocr, japanese=reviewer
    )
    assert ocr.sizes == [(500, 300), (500, 300)]
    assert ocr.api.requests == 2 and reviewer.api.requests == 1
    assert len(result.pages) == 2 and result.japanese_review_completed
    assert result.status == "checks_passed"
    assert structure.calls == [
        (stage, (500, 300)) for _ in range(2) for stage in ("detect", "recognize")
    ]
    assert (
        Result.model_validate_json((directory / "simple.json").read_text(encoding="utf-8"))
        == result
    )
    assert result.pages[0].tables[0].cells[-1].text == "2,000"
    assert result.pages[0].tables[0].cells[-1].bbox == (250, 150, 450, 270)
    assert result.pages[0].tables[0].geometry_valid
    assert result.pages[0].tables[0].rows[0].bbox == (50, 30, 450, 150)
    assert result.pages[0].annotated_image == "pages/p001-annotated.png"
    assert (directory / "pages/p001-annotated.png").is_file()
    assert (directory / "result.json").is_file()
    schema = json.loads((directory / "simple.schema.json").read_text(encoding="utf-8"))
    assert schema["title"] == "Result"
    for page in result.pages:
        with Image.open(directory / page.image) as image:
            assert image.format == "PNG" and image.size == (500, 300)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert all(
        sha256(directory / name) == digest for name, digest in manifest["artifact_sha256"].items()
    )


def test_image_input_is_uncropped_and_transparency_is_composited_on_white(
    config, tmp_path, monkeypatch
):
    source = tmp_path / "transparent.png"
    with Image.new("RGBA", (80, 40), (255, 0, 0, 128)) as image:
        image.save(source)

    def forbid_crop(*args, **kwargs):
        raise AssertionError("No cropping allowed")

    monkeypatch.setattr(Image.Image, "crop", forbid_crop)
    directory, result = run(
        source, config, tmp_path / "output", structure=Structure(), ocr=OCR(), japanese=Reviewer()
    )
    with Image.open(directory / result.pages[0].image) as image:
        assert image.size == (80, 40) and image.mode == "RGB"
        assert image.getpixel((0, 0)) == (255, 127, 127)


def test_provider_failure_keeps_png_and_exports_failure_json(config, pdf, tmp_path):
    directory, result = run(
        pdf,
        config,
        tmp_path / "output",
        structure=Structure(),
        ocr=OCR(error=True),
        japanese=Reviewer(),
    )
    assert result.status == "needs_review"
    assert any(i.code == "page_alignment_uncertain" for i in result.issues)
    assert (directory / result.pages[0].image).is_file()
    assert (directory / "result.json").is_file()


def test_oversized_image_is_rejected_before_api_calls(config, tmp_path):
    source = tmp_path / "large.png"
    with Image.new("RGB", (50, 50)) as image:
        image.save(source)
    config.render.max_pixels = 1000
    ocr = OCR()
    _, result = run(
        source, config, tmp_path / "output", structure=Structure(), ocr=ocr, japanese=Reviewer()
    )
    assert result.status == "failed"
    assert any("pixel limit" in i.message for i in result.issues)
    assert ocr.api.requests == 0


def test_multi_frame_image_is_rejected_instead_of_dropping_frames(config, tmp_path):
    source = tmp_path / "two.tiff"
    with Image.new("RGB", (40, 40)) as first, Image.new("RGB", (40, 40)) as second:
        first.save(source, save_all=True, append_images=[second])
    _, result = run(
        source, config, tmp_path / "output", structure=Structure(), ocr=OCR(), japanese=Reviewer()
    )
    assert result.status == "failed"
    assert any("Multi-frame" in i.message for i in result.issues)


def test_config_and_env_paths_work_outside_repo():
    root = Path(__file__).resolve().parents[1]
    settings = load_config(root / "config.toml", "table")
    assert settings.env_file == root / ".env"
    assert settings.detection.local_dir == root / "models/tatr/detection"
    assert settings.structure.local_dir == root / "models/tatr/structure"
    assert settings.japanese.max_requests == 1 and settings.japanese.format_retries == 0
