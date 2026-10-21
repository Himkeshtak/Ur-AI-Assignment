"""Command-line entry point. No model weights are loaded for schema or doctor."""

import argparse
import importlib.util
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from .api import APIError, credential
from .artifacts import write_json
from .config import Config, GroqSettings, JapaneseSettings, OCRSettings, VLMRunSettings
from .models import Document


def doctor(config: Config) -> bool:
    required = {
        "pydantic": "pydantic",
        "pillow": "PIL",
        "pypdfium2": "pypdfium2",
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
        "timm": "timm",
        "httpx": "httpx",
        "python-dotenv": "dotenv",
    }
    if isinstance(config.ocr, OCRSettings):
        required.update({"paddleocr": "paddleocr", "paddlepaddle": "paddle"})
    if isinstance(config.japanese, JapaneseSettings):
        required["sentencepiece"] = "sentencepiece"
    packages = {}
    for package, module in required.items():
        try:
            installed = version(package)
        except PackageNotFoundError:
            installed = None
        packages[package] = {
            "version": installed,
            "available": importlib.util.find_spec(module) is not None,
        }
    ready = all(package["available"] for package in packages.values())
    credentials = {}
    for spec in (config.ocr, config.japanese):
        if isinstance(spec, (VLMRunSettings, GroqSettings)):
            try:
                credential(spec.api_key_env, config.env_file)
                credentials[spec.api_key_env] = True
            except APIError:
                credentials[spec.api_key_env] = False
    print(
        json.dumps(
            {
                "dependencies_present": ready,
                "packages": packages,
                "credentials_present": credentials,
                "backends": {"ocr": config.ocr.backend, "japanese": config.japanese.backend},
                "model_revisions": {
                    name: getattr(getattr(config, name), "revision", None)
                    for name in (
                        "detection",
                        "structure",
                        "ocr",
                        "japanese",
                    )
                },
                "note": "Presence only; credentials, models, and inference are not verified here.",
            },
            indent=2,
        )
    )
    return ready and all(credentials.values())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parse Japanese PDF tables with explicit geometry")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run the approved extraction and validation pipeline")
    run.add_argument("pdf", type=Path)
    run.add_argument("--config", type=Path, default=Path("config.toml"))
    run.add_argument("--output-dir", type=Path, default=Path("output"))
    check = commands.add_parser("doctor", help="Check configuration and dependency presence")
    check.add_argument("--config", type=Path, default=Path("config.toml"))
    download = commands.add_parser(
        "download-tatr", help="Download and checksum the official TATR assets"
    )
    download.add_argument("--config", type=Path, default=Path("config.toml"))
    tatr = commands.add_parser("check-tatr", help="Run both TATR checkpoints locally on CPU")
    tatr.add_argument("--config", type=Path, default=Path("config.toml"))
    tatr.add_argument("--output-dir", type=Path, default=Path("output/tatr-check"))
    tatr.add_argument(
        "--image", type=Path, help="Optional page image; defaults to a synthetic table"
    )
    api = commands.add_parser("check-api", help="Make small live calls on synthetic inputs")
    api.add_argument("--config", type=Path, default=Path("config.toml"))
    api.add_argument("--output-dir", type=Path, default=Path("output/api-check"))
    schema = commands.add_parser("schema", help="Export the application's JSON Schema")
    schema.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "schema":
            write_json(args.output, Document.model_json_schema())
            print(args.output)
            return 0
        config = Config.load(args.config)
        if args.command == "doctor":
            return 0 if doctor(config) else 1
        if args.command == "check-api":
            from .check_api import check_api

            passed, directory = check_api(config, args.output_dir)
            print(directory / "report.json")
            return 0 if passed else 1
        if args.command in ("download-tatr", "check-tatr"):
            from .checkpoints import check_tatr, download_tatr

            if args.command == "download-tatr":
                print(json.dumps(download_tatr(config), indent=2))
            else:
                check_tatr(config, args.output_dir, args.image)
                print(args.output_dir / "report.json")
            return 0
        from .pipeline import Pipeline

        result = Pipeline(config).run(args.pdf, args.output_dir)
        print(
            json.dumps({"status": result.document.status, "run_directory": str(result.directory)})
        )
        return {"checks_passed": 0, "needs_review": 2, "failed": 1}[result.document.status]
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
