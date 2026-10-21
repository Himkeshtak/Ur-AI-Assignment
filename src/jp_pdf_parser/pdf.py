"""Sequential, bounded PDF rendering and invertible table crops."""

import math
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image

from .artifacts import Artifacts, sha256
from .config import RenderSettings
from .models import Affine, Detection, Page


def inspect_pdf(path: Path, settings: RenderSettings) -> tuple[str, int]:
    if not path.is_file():
        raise ValueError(f"PDF does not exist: {path}")
    if path.stat().st_size > settings.max_file_mb * 1024 * 1024:
        raise ValueError("PDF exceeds configured file size limit")
    with path.open("rb") as stream:
        if b"%PDF-" not in stream.read(1024):
            raise ValueError("Input does not have a PDF signature")
    with pdfium.PdfDocument(path) as document:
        count = len(document)
        if not 1 <= count <= settings.max_pages:
            raise ValueError("PDF page count is outside configured limits")
    return sha256(path), count


def render_pages(path: Path, settings: RenderSettings, artifacts: Artifacts) -> Iterator[Page]:
    """Keep one raster in memory; reject oversized pages before allocating a bitmap."""
    with pdfium.PdfDocument(path) as document:
        for index in range(len(document)):
            with artifacts.stage(f"B.render.p{index + 1}"), closing(document[index]) as page:
                width, height = page.get_size()
                scale = settings.dpi / 72
                if math.ceil(width * scale) * math.ceil(height * scale) > settings.max_pixels:
                    raise ValueError(f"Page {index + 1} exceeds configured pixel limit")
                bitmap = page.render(scale=scale, fill_color=(255, 255, 255, 255))
                try:
                    converter = bitmap.get_posconv(page)
                    origin = converter.to_page(0, 0)
                    x_axis = converter.to_page(1, 0)
                    y_axis = converter.to_page(0, 1)
                    transform = (
                        x_axis[0] - origin[0],
                        y_axis[0] - origin[0],
                        origin[0],
                        x_axis[1] - origin[1],
                        y_axis[1] - origin[1],
                        origin[1],
                    )
                    image = bitmap.to_pil().convert("RGB")
                    relative = f"pages/p{index + 1:03d}.png"
                    image.save(artifacts.path(relative))
                    result = Page(
                        number=index + 1,
                        width_points=width,
                        height_points=height,
                        rotation=page.get_rotation(),
                        width_pixels=image.width,
                        height_pixels=image.height,
                        dpi=settings.dpi,
                        image_to_pdf=transform,
                        image_path=relative,
                    )
                    image.close()
                finally:
                    bitmap.close()
            yield result


def crop_table(
    image: Image.Image,
    detection: Detection,
    padding: int,
) -> tuple[Image.Image, Affine]:
    x0, y0, x1, y1 = detection.bbox
    left = max(0, math.floor(x0) - padding)
    top = max(0, math.floor(y0) - padding)
    right = min(image.width, math.ceil(x1) + padding)
    bottom = min(image.height, math.ceil(y1) + padding)
    if right <= left or bottom <= top:
        raise ValueError("Table detection lies outside the page")
    crop = image.crop((left, top, right, bottom))
    if detection.label == "table rotated":
        height = crop.height
        rotated = crop.transpose(Image.Transpose.ROTATE_270)
        crop.close()
        return rotated, (0, 1, left, -1, 0, top + height)
    return crop, (1, 0, left, 0, 1, top)
