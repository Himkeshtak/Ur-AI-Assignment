"""Opt-in live checks on a tiny synthetic table; never runs during tests/doctor."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .artifacts import Artifacts
from .config import Config, GroqSettings, VLMRunSettings
from .grid import numbered_overlay, reconstruct_grid
from .japanese import GroqReviewer
from .models import Detection, Document, Page
from .ocr import VLMRunOCR


def check_api(config: Config, output_dir: Path) -> tuple[bool, Path]:
    if not isinstance(config.ocr, VLMRunSettings) or not isinstance(config.japanese, GroqSettings):
        raise ValueError("check-api requires the vlmrun and groq backends")
    ocr = VLMRunOCR(config.ocr, config.env_file)
    reviewer = GroqReviewer(config.japanese, config.env_file)
    ocr.api.check_credentials()
    reviewer.api.check_credentials()
    artifacts = Artifacts(output_dir)
    page = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default(size=30)
    values = (("Item", "Amount"), ("Paper", "2,000"))
    for r, row in enumerate(values):
        for c, text in enumerate(row):
            draw.rectangle(
                (20 + c * 380, 60 + r * 140, 400 + c * 380, 200 + r * 140), outline="black", width=2
            )
            draw.text((40 + c * 380, 110 + r * 140), text, font=font, fill="black")
    page.save(artifacts.path("table.png"))
    with page.crop((405, 205, 775, 335)) as cell:
        cell.save(artifacts.path("cell.png"))
    objects = [Detection(label="table", score=1, bbox=(20, 60, 780, 340))]
    objects += [
        Detection(label="table row", score=1, bbox=(20, 60 + r * 140, 780, 200 + r * 140))
        for r in range(2)
    ]
    objects += [
        Detection(label="table column", score=1, bbox=(20 + c * 380, 60, 400 + c * 380, 340))
        for c in range(2)
    ]
    table = reconstruct_grid(
        "p1-t1",
        1,
        "table.png",
        page.size,
        (1, 0, 0, 0, 1, 0),
        objects,
        config.structure.threshold,
        10,
    )
    with numbered_overlay(page, table, config.render.max_pixels) as overlay:
        overlay.save(artifacts.path("numbered.png"))
    page.close()
    results = {}
    for name, filename, method in (
        ("clean_cell", "cell.png", "ocr"),
        ("annotated_table", "numbered.png", "table"),
    ):
        try:
            prediction = ocr.recognize(artifacts.path(filename), method)
            artifacts.write(f"{name}.json", prediction.raw)
            artifacts.write(f"{name}-text.json", {"text": prediction.text})
            results[name] = {"completed": True, "has_text": bool(prediction.text.strip())}
            if name == "clean_cell":
                results[name]["comma_preserved"] = prediction.text.strip() == "2,000"
            else:
                results[name]["html_available"] = prediction.text.lstrip().startswith("<table")
        except Exception as exc:
            results[name] = {"completed": False, "error": str(exc)}
    # This independent Japanese probe tests review format/coverage, not visual recognition.
    for cell, text in zip(table.cells, ("品目", "金額", "用紙", "2,000"), strict=True):
        cell.raw_text = text
    document = Document(
        source_name="synthetic-api-check",
        page_count=1,
        pages=[
            Page(
                number=1,
                width_points=800,
                height_points=400,
                rotation=0,
                width_pixels=800,
                height_pixels=400,
                dpi=72,
                image_to_pdf=(1, 0, 0, 0, 1, 0),
                image_path="table.png",
                tables=[table],
            )
        ],
    )
    try:
        run = reviewer.review(document)
        artifacts.write("japanese-review.json", run.report)
        artifacts.write("japanese-traces.json", run.traces)
        results["japanese"] = {
            "completed": run.report.completed,
            "reviewed_cells": len(run.report.reviewed_cell_ids),
        }
    except Exception as exc:
        results["japanese"] = {"completed": False, "error": str(exc)}
    passed = all(result["completed"] for result in results.values())
    artifacts.write(
        "report.json",
        {
            "transport_checks_passed": passed,
            "accuracy_evaluated": False,
            "results": results,
            "requests": {"vlmrun": ocr.api.requests, "groq": reviewer.api.requests},
        },
    )
    artifacts.finish(
        config,
        None,
        {"ocr": "jp_pdf_parser.ocr.VLMRunOCR", "japanese": "jp_pdf_parser.japanese.GroqReviewer"},
    )
    return passed, artifacts.root
