"""Application contracts. JSON Schema is generated from these models."""

from math import isfinite
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


def valid_box(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = value
    if not all(isfinite(x) for x in value) or x0 >= x1 or y0 >= y1:
        raise ValueError("Bounding boxes must be finite and have positive area")
    return value


Box = Annotated[tuple[float, float, float, float], AfterValidator(valid_box)]
# x' = a*x + b*y + c; y' = d*x + e*y + f. Pixel bounds are half-open.
Affine = tuple[float, float, float, float, float, float]
Status = Literal["checks_passed", "needs_review", "failed"]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class Issue(Record):
    code: str
    stage: str
    message: str
    cell_ids: list[str] = Field(default_factory=list)
    severity: Literal["warning", "error"] = "warning"


class Detection(Record):
    label: str
    score: float = Field(ge=0, le=1)
    bbox: Box


class Axis(Record):
    id: str
    index: int = Field(ge=1)
    bbox: Box
    coordinate_frame: Literal["upright_table_pixels", "rendered_page_pixels"] = (
        "upright_table_pixels"
    )
    score: float = Field(ge=0, le=1)
    is_header: bool = False


class Reading(Record):
    request_id: str
    variant: Literal["clean_cell", "annotated_table", "annotated_page"]
    image_path: str
    raw_path: str | None = None
    text: str | None = None
    error: str | None = None
    source_bbox_in_image: Box | None = None
    image_to_page: Affine | None = None


class Cell(Record):
    id: str
    row: int = Field(ge=1)
    column: int = Field(ge=1)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    bbox: Box
    table_bbox: Box
    coordinate_frame: Literal["rendered_page_pixels"] = "rendered_page_pixels"
    bbox_source: Literal["tatr_grid_reconstruction"] = "tatr_grid_reconstruction"
    row_ids: list[str]
    column_ids: list[str]
    is_header: bool = False
    raw_text: str | None = None
    normalized_text: str | None = None
    number: str | None = None
    readings: list[Reading] = Field(default_factory=list)


class InvoiceLine(Record):
    row: int
    amount_id: str
    quantity_id: str | None = None
    unit_price_id: str | None = None


class InvoiceView(Record):
    header_row: int
    columns: dict[str, int]
    lines: list[InvoiceLine] = Field(default_factory=list)
    subtotal_id: str | None = None
    total_id: str | None = None
    tax_ids: list[str] = Field(default_factory=list)
    adjustment_ids: list[str] = Field(default_factory=list)


class Table(Record):
    id: str
    page_number: int
    grid_revision: int = 1
    bbox: Box
    coordinate_frame: Literal["rendered_page_pixels"] = "rendered_page_pixels"
    table_to_page: Affine
    structure_coordinate_frame: Literal["upright_table_pixels", "rendered_page_pixels"] = (
        "upright_table_pixels"
    )
    image_path: str
    rows: list[Axis] = Field(default_factory=list)
    columns: list[Axis] = Field(default_factory=list)
    cells: list[Cell] = Field(default_factory=list)
    geometry_valid: bool = False
    issues: list[Issue] = Field(default_factory=list)
    annotated_reading: Reading | None = None
    invoice: InvoiceView | None = None


class Page(Record):
    number: int
    width_points: float
    height_points: float
    rotation: int
    width_pixels: int
    height_pixels: int
    dpi: int
    image_to_pdf: Affine
    image_path: str
    annotated_reading: Reading | None = None
    tables: list[Table] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class Check(Record):
    rule: str
    status: Literal["passed", "failed", "incomplete", "not_applicable"]
    cell_ids: list[str] = Field(default_factory=list)
    expected: str | None = None
    observed: str | None = None
    difference: str | None = None
    detail: str


class JapaneseReview(Record):
    completed: bool = False
    reviewed_cell_ids: list[str] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)


class Document(Record):
    schema_version: str = "1.1.0"
    source_name: str
    source_sha256: str | None = None
    page_count: int = 0
    pages: list[Page] = Field(default_factory=list)
    status: Status = "failed"
    checks: list[Check] = Field(default_factory=list)
    issues: list[Issue] = Field(default_factory=list)
    japanese_review: JapaneseReview | None = None

    def all_cells(self) -> list[Cell]:
        return [cell for page in self.pages for table in page.tables for cell in table.cells]

    def all_issues(self) -> list[Issue]:
        return [
            *self.issues,
            *(issue for page in self.pages for issue in page.issues),
            *(issue for page in self.pages for table in page.tables for issue in table.issues),
            *(self.japanese_review.issues if self.japanese_review else []),
        ]
