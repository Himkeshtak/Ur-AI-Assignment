"""Download and verify the exact official TATR assets used by the CPU adapter."""

import json
import time
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from PIL import Image, ImageDraw

from .artifacts import sha256, write_json
from .config import Config, ObjectSettings
from .pdf import crop_table


def checkpoint_entry(spec: ObjectSettings) -> dict:
    catalog = json.loads(
        files("jp_pdf_parser").joinpath("tatr_checkpoints.json").read_text("utf-8")
    )
    for entry in catalog.values():
        if (entry["model_id"], entry["revision"]) == (spec.model_id, spec.revision):
            return entry
    raise ValueError(
        f"No verified local TATR checkpoint registered for {spec.model_id}@{spec.revision}"
    )


def verify_checkpoint(spec: ObjectSettings) -> dict:
    """Verify every locked asset before local model loading. This never uses the network."""
    if spec.local_dir is None:
        raise ValueError("Set local_dir for both TATR checkpoints before downloading/checking them")
    entry = checkpoint_entry(spec)
    for filename, expected in entry["files"].items():
        path = spec.local_dir / filename
        if not path.is_file():
            raise ValueError(f"Missing TATR asset: {path}. Run download-tatr first.")
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"TATR checksum mismatch: {path}. Run download-tatr to restore it.")
    return {**entry, "local_dir": str(spec.local_dir)}


def download_tatr(config: Config) -> dict:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("Install the 'tatr' extra to download TATR") from exc
    result = {}
    for name in ("detection", "structure"):
        spec = getattr(config, name)
        if spec.local_dir is None:
            raise ValueError(f"Set {name}.local_dir before downloading TATR")
        entry = checkpoint_entry(spec)
        for filename, expected in entry["files"].items():
            path = spec.local_dir / filename
            if path.is_file():
                if path.stat().st_size == expected["bytes"] and sha256(path) == expected["sha256"]:
                    continue
            hf_hub_download(
                repo_id=spec.model_id,
                revision=spec.revision,
                filename=filename,
                local_dir=spec.local_dir,
                cache_dir=spec.cache_dir,
                token=False,  # These Microsoft checkpoints are public; no credential is needed.
                force_download=path.exists(),
            )
        result[name] = verify_checkpoint(spec)
    return result


def check_tatr(config: Config, output_dir: Path, image_path: Path | None = None) -> dict:
    """Run both real checkpoints offline; synthetic input checks execution, not accuracy."""
    import torch

    from .structure import TATR

    for name in ("detection", "structure"):
        spec = getattr(config, name)
        if spec.device != "cpu":
            raise ValueError("check-tatr requires both TATR devices to be cpu")
        verify_checkpoint(spec)
    output_dir.mkdir(parents=True, exist_ok=True)
    if image_path is None:
        image = Image.new("RGB", (800, 600), "white")
        draw = ImageDraw.Draw(image)
        for row, values in enumerate(
            (
                ("Item", "Quantity", "Unit price", "Amount"),
                ("Paper", "2", "1,000", "2,000"),
                ("Pens", "3", "100", "300"),
                ("Total", "", "", "2,300"),
            )
        ):
            for column, value in enumerate(values):
                draw.rectangle(
                    (40 + column * 180, 80 + row * 80, 220 + column * 180, 160 + row * 80),
                    outline="black",
                    width=2,
                )
                draw.text((50 + column * 180, 110 + row * 80), value, fill="black")
        source = "synthetic English table; execution check only"
    else:
        with Image.open(image_path) as original:
            image = original.convert("RGB")
        source = str(image_path)
    image.save(output_dir / "input.png")
    backend = TATR(config.detection, config.structure)
    report = {
        "source": source,
        "accuracy_evaluated": False,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "packages": {
            name: version(name) for name in ("torch", "torchvision", "transformers", "timm")
        },
        "input_sha256": sha256(output_dir / "input.png"),
        "models": {},
    }
    detections = []
    transform = (1, 0, 0, 0, 1, 0)
    for name in ("detection", "structure"):
        spec = getattr(config, name)
        started = time.perf_counter()
        processor, _ = backend.load(spec)
        load_seconds = time.perf_counter() - started
        # For a real page, structure recognition receives the highest-scoring table crop.
        if name == "structure":
            tables = [
                d
                for d in detections
                if d.label in ("table", "table rotated") and d.score >= config.detection.threshold
            ]
            if tables:
                table = max(tables, key=lambda d: d.score)
                crop, transform = crop_table(image, table, config.render.table_padding)
                image.close()
                image = crop
                source_frame = "detected table crop"
            else:
                source_frame = "whole input; no table passed the detection threshold"
            image.save(output_dir / "structure-input.png")
        else:
            source_frame = "input image"
        started = time.perf_counter()
        detections = backend.detect(image) if name == "detection" else backend.recognize(image)
        inference_seconds = time.perf_counter() - started
        report["models"][name] = {
            "checkpoint": verify_checkpoint(spec),
            "load_seconds": load_seconds,
            "inference_seconds": inference_seconds,
            "input_frame": source_frame,
            "input_size": list(image.size),
            "image_to_page": list(transform),
            "processor_size": processor.size,
            "objects_above_threshold": sum(d.score >= spec.threshold for d in detections),
            "raw_objects": [d.model_dump(mode="json") for d in detections],
        }
    image.close()
    write_json(output_dir / "report.json", report)
    return report
