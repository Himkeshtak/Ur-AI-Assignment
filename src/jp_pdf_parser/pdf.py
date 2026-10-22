"""Sequential, bounded PDF rendering; page images remain intact."""

import math
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image, ImageOps

from .artifacts import Artifacts, sha256
from .config import RenderSettings
from .models import Page

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def inspect_source(path: Path, settings: RenderSettings) -> tuple[str, int]:
    if path.suffix.lower() == ".pdf":
        return inspect_pdf(path, settings)
    if not path.is_file():
        raise ValueError(f"Input does not exist: {path}")
    if path.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Use a PDF, PNG, JPEG, BMP, single-frame TIFF or WebP file")
    if path.stat().st_size > settings.max_file_mb * 1024 * 1024:
        raise ValueError("Input exceeds the configured file-size limit")
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are unsupported; supply a PDF or single image")
        if image.width * image.height > settings.max_pixels:
            raise ValueError("Image exceeds the configured pixel limit")
    return sha256(path), 1


def render_source(path: Path, settings: RenderSettings, artifacts: Artifacts) -> Iterator[Page]:
    if path.suffix.lower() == ".pdf":
        yield from render_pages(path, settings, artifacts)
        return
    with artifacts.stage("B.normalize-image"), Image.open(path) as original:
        with ImageOps.exif_transpose(original) as oriented:
            with oriented.convert("RGBA") as rgba, Image.new("RGB", rgba.size, "white") as image:
                with rgba.getchannel("A") as alpha:
                    image.paste(rgba, mask=alpha)
                image.save(artifacts.path("pages/p001.png"))
                # Raster inputs use a nominal 72-DPI virtual page, with pixel coordinates.
                page = Page(
                    number=1,
                    width_points=image.width,
                    height_points=image.height,
                    rotation=0,
                    width_pixels=image.width,
                    height_pixels=image.height,
                    dpi=72,
                    image_to_pdf=(1, 0, 0, 0, 1, 0),
                    image_path="pages/p001.png",
                )
    yield page


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
