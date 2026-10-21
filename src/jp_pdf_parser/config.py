"""Typed, reproducible configuration. Model loading is always explicit."""

import re
import tomllib
from decimal import Decimal
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class ModelSettings(Settings):
    model_id: str
    revision: str
    cache_dir: str = ".cache/huggingface"
    device: str = "cpu"

    @field_validator("revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-f]{40}", value):
            raise ValueError("Use an immutable 40-character checkpoint commit ID")
        return value


class ObjectSettings(ModelSettings):
    threshold: float = Field(default=0.6, ge=0, le=1)
    local_dir: Path | None = None


class RenderSettings(Settings):
    dpi: int = Field(default=300, ge=72, le=600)
    max_pages: int = Field(default=30, ge=1)
    max_file_mb: int = Field(default=50, ge=1)
    max_pixels: int = Field(default=24_000_000, ge=1000)
    max_cells_per_table: int = Field(default=500, ge=1)
    table_padding: int = Field(default=12, ge=0, le=100)


class OCRSettings(ModelSettings):
    max_new_tokens: int = Field(default=1024, ge=32)
    max_pixels: int = Field(default=1_605_632, ge=1024)
    max_retries_per_table: int = Field(default=1, ge=0, le=1)


class JapaneseSettings(ModelSettings):
    context_tokens: int = Field(default=4096, ge=1024, le=4096)
    max_new_tokens: int = Field(default=512, ge=128, le=1024)
    format_retries: int = Field(default=1, ge=0, le=1)


class FinancialSettings(Settings):
    mode: Literal["invoice", "table"] = "invoice"
    tax_mode: Literal["unknown", "exclusive", "inclusive"] = "unknown"
    money_scale: int = Field(default=0, ge=0, le=4)
    rounding: Literal["ROUND_HALF_UP", "ROUND_DOWN", "ROUND_UP"] | None = None
    tax_rate: Decimal | None = Field(default=None, ge=0, le=1)


class Config(Settings):
    render: RenderSettings = Field(default_factory=RenderSettings)
    detection: ObjectSettings
    structure: ObjectSettings
    ocr: OCRSettings
    japanese: JapaneseSettings
    financial: FinancialSettings = Field(default_factory=FinancialSettings)

    @classmethod
    def load(cls, path: Path) -> "Config":
        with path.open("rb") as stream:
            return cls.model_validate(tomllib.load(stream))
