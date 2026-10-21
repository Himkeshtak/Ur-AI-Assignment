"""Local checkpoint integrity, offline loading, and processor compatibility."""

import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

from jp_pdf_parser import checkpoints
from jp_pdf_parser.structure import TATR


@pytest.fixture
def local_checkpoint(config, tmp_path, monkeypatch):
    payloads = {
        "config.json": b'{"model_type":"table-transformer"}',
        "preprocessor_config.json": b'{"size":{"longest_edge":800}}',
        "model.safetensors": b"test weights; not a real checkpoint",
        "README.md": b"test card",
    }
    for name in ("detection", "structure"):
        spec = getattr(config, name)
        spec.local_dir = tmp_path / name
        spec.local_dir.mkdir()
        for filename, content in payloads.items():
            (spec.local_dir / filename).write_bytes(content)

    def entry(spec):
        return {
            "model_id": spec.model_id,
            "revision": spec.revision,
            "files": {
                filename: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                for filename, content in payloads.items()
            },
        }

    monkeypatch.setattr(checkpoints, "checkpoint_entry", entry)
    return config, payloads


@pytest.mark.parametrize("damage", ["missing", "truncated", "same_size_corruption"])
def test_local_checkpoint_rejects_damaged_weights(local_checkpoint, damage):
    config, payloads = local_checkpoint
    spec = config.detection
    checkpoints.verify_checkpoint(spec)
    path = spec.local_dir / "model.safetensors"
    if damage == "missing":
        path.unlink()
    elif damage == "truncated":
        path.write_bytes(b"short")
    else:
        path.write_bytes(b"x" * len(payloads["model.safetensors"]))
    with pytest.raises(ValueError, match="Missing TATR asset|checksum mismatch"):
        checkpoints.verify_checkpoint(spec)


def test_catalog_requires_exact_revision(config):
    assert checkpoints.checkpoint_entry(config.detection)["revision"] == config.detection.revision
    config.detection.revision = "0" * 40
    with pytest.raises(ValueError, match="No verified local TATR checkpoint"):
        checkpoints.checkpoint_entry(config.detection)


def test_download_skips_valid_files_and_repairs_corruption(local_checkpoint, monkeypatch):
    config, payloads = local_checkpoint
    hub = ModuleType("huggingface_hub")
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        (kwargs["local_dir"] / kwargs["filename"]).write_bytes(payloads[kwargs["filename"]])

    hub.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    checkpoints.download_tatr(config)
    assert not calls
    (config.structure.local_dir / "model.safetensors").write_bytes(b"broken")
    (config.detection.local_dir / "config.json").unlink()
    result = checkpoints.download_tatr(config)
    assert set(result) == {"detection", "structure"}
    assert len(calls) == 2
    assert calls[0]["filename"] == "config.json"
    assert calls[0]["force_download"] is False
    assert calls[0]["revision"] == config.detection.revision
    assert calls[1]["filename"] == "model.safetensors"
    assert calls[1]["force_download"] is True
    assert calls[1]["token"] is False


def test_loader_is_offline_and_adapts_longest_edge_without_changing_files(
    local_checkpoint, monkeypatch
):
    config, payloads = local_checkpoint
    calls = []

    def processor(source, **kwargs):
        calls.append(("processor", source, kwargs))
        return SimpleNamespace(size={"longest_edge": 800})

    model = SimpleNamespace(to=lambda _: model, eval=lambda: model)

    def load_model(source, **kwargs):
        calls.append(("model", source, kwargs))
        return model

    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = SimpleNamespace(from_pretrained=processor)
    transformers.AutoModelForObjectDetection = SimpleNamespace(from_pretrained=load_model)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    backend = TATR(config.detection, config.structure)
    result, _ = backend.load(config.structure)
    assert result.size == {"shortest_edge": 800, "longest_edge": 800}
    assert (config.structure.local_dir / "preprocessor_config.json").read_bytes() == payloads[
        "preprocessor_config.json"
    ]
    assert all(kwargs["local_files_only"] for _, _, kwargs in calls)
    assert calls[1][2]["use_pretrained_backbone"] is False
    assert calls[1][2]["use_safetensors"] is True
    backend.load(config.structure)
    assert len(calls) == 2


def test_loader_rejects_corruption_before_transformer_loading(local_checkpoint, monkeypatch):
    config, _ = local_checkpoint
    (config.detection.local_dir / "model.safetensors").unlink()
    transformers = ModuleType("transformers")
    transformers.AutoImageProcessor = SimpleNamespace()
    transformers.AutoModelForObjectDetection = SimpleNamespace()
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(ValueError, match="Missing TATR asset"):
        TATR(config.detection, config.structure).load(config.detection)
