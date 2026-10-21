"""Atomic output writes and run provenance; no invoice text is logged."""

import hashlib
import json
import platform
import subprocess
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


class Artifacts:
    def __init__(self, root: Path):
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.root = root / f"run-{stamp}-{uuid4().hex[:12]}"
        self.root.mkdir(parents=True, exist_ok=False)
        self.started_at = datetime.now(UTC).isoformat()
        self.timings: list[dict] = []

    def path(self, relative: str) -> Path:
        target = self.root / relative
        if not target.resolve().is_relative_to(self.root.resolve()):
            raise ValueError("Artifact path leaves the run directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        return target

    def write(self, relative: str, value: object) -> None:
        write_json(self.path(relative), value)

    @contextmanager
    def stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timings.append({"stage": name, "seconds": time.perf_counter() - started})

    def finish(self, config: BaseModel, source_hash: str | None, backends: dict[str, str]) -> None:
        packages = {}
        for package in (
            "jp-pdf-parser",
            "pydantic",
            "pillow",
            "pypdfium2",
            "httpx",
            "python-dotenv",
            "torch",
            "transformers",
            "paddleocr",
            "paddlex",
            "paddlepaddle",
        ):
            try:
                packages[package] = version(package)
            except PackageNotFoundError:
                packages[package] = None
        repository = Path(__file__).resolve().parents[2]
        try:
            if not (repository / ".git").exists():
                raise OSError("Installed wheel has no source repository")
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
                timeout=5,
                cwd=repository,
            ).stdout.strip()
            dirty = bool(
                subprocess.run(
                    ["git", "status", "--porcelain"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=5,
                    cwd=repository,
                ).stdout
            )
        except (OSError, subprocess.SubprocessError):
            revision, dirty = None, None
        digests = {
            path.relative_to(self.root).as_posix(): sha256(path)
            for path in sorted(self.root.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        self.write(
            "manifest.json",
            {
                "started_at": self.started_at,
                "finished_at": datetime.now(UTC).isoformat(),
                "source_sha256": source_hash,
                "application_revision": revision,
                "working_tree_dirty": dirty,
                "python": platform.python_version(),
                "platform": platform.platform(),
                "packages": packages,
                "config": config.model_dump(mode="json"),
                "backends": backends,
                "application_source_sha256": {
                    path.name: sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))
                },
                "dependency_lock_sha256": (
                    sha256(repository / "uv.lock") if (repository / "uv.lock").is_file() else None
                ),
                "timings": self.timings,
                "artifact_sha256": digests,
            },
        )
