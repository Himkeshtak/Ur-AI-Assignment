"""Reconstruct predicted table topology without relying on VLM row assignments."""

from PIL import Image, ImageDraw, ImageFont

from .models import Affine, Axis, Box, Cell, Detection, Issue, Table


def transform_box(box: Box, transform: Affine) -> Box:
    a, b, c, d, e, f = transform
    points = [
        (a * x + b * y + c, d * x + e * y + f) for x in (box[0], box[2]) for y in (box[1], box[3])
    ]
    return (
        min(p[0] for p in points),
        min(p[1] for p in points),
        max(p[0] for p in points),
        max(p[1] for p in points),
    )


def intersection_area(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def area(box: Box) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def suppress_duplicates(objects: list[Detection], overlap: float = 0.6) -> list[Detection]:
    kept: list[Detection] = []
    for obj in sorted(objects, key=lambda obj: (-obj.score, obj.bbox)):
        if not any(
            intersection_area(obj.bbox, other.bbox) / min(area(obj.bbox), area(other.bbox))
            > overlap
            for other in kept
        ):
            kept.append(obj)
    return kept


def _boundaries(objects: list[Detection], start: float, end: float, axis: int) -> list[float]:
    return [
        start,
        *[
            (left.bbox[axis + 2] + right.bbox[axis]) / 2
            for left, right in zip(objects, objects[1:], strict=False)
        ],
        end,
    ]


def reconstruct_grid(
    table_id: str,
    page_number: int,
    image_path: str,
    size: tuple[int, int],
    table_to_page: Affine,
    objects: list[Detection],
    threshold: float,
    max_cells: int,
    *,
    region: Box | None = None,
) -> Table:
    """Refine axes, intersect them, then replace covered positions with merged cells."""
    width, height = size
    bounds = region or (0, 0, width, height)
    frame = "rendered_page_pixels" if region is not None else "upright_table_pixels"
    table = Table(
        id=table_id,
        page_number=page_number,
        image_path=image_path,
        bbox=transform_box(bounds, table_to_page),
        structure_coordinate_frame=frame,
        table_to_page=table_to_page,
    )
    candidates = []
    for obj in objects:
        if obj.score < threshold:
            continue
        if region is not None:
            cx, cy = (obj.bbox[0] + obj.bbox[2]) / 2, (obj.bbox[1] + obj.bbox[3]) / 2
            if not (bounds[0] <= cx <= bounds[2] and bounds[1] <= cy <= bounds[3]):
                continue
            if intersection_area(obj.bbox, bounds) / area(obj.bbox) < 0.5:
                continue
            obj = obj.model_copy(
                update={
                    "bbox": (
                        max(bounds[0], obj.bbox[0]),
                        max(bounds[1], obj.bbox[1]),
                        min(bounds[2], obj.bbox[2]),
                        min(bounds[3], obj.bbox[3]),
                    )
                }
            )
        candidates.append(obj)
    rows = sorted(
        suppress_duplicates([o for o in candidates if o.label == "table row"]),
        key=lambda o: (o.bbox[1], o.bbox[0]),
    )
    columns = sorted(
        suppress_duplicates([o for o in candidates if o.label == "table column"]),
        key=lambda o: (o.bbox[0], o.bbox[1]),
    )
    if not rows or not columns:
        table.issues.append(
            Issue(
                code="missing_axes",
                stage="E",
                severity="error",
                message="Structure recognition did not establish rows and columns",
            )
        )
        return table
    if len(rows) * len(columns) > max_cells:
        table.issues.append(
            Issue(
                code="cell_limit",
                stage="E",
                severity="error",
                message="Predicted grid exceeds configured cell limit",
            )
        )
        return table
    x0 = max(bounds[0], min(o.bbox[0] for o in columns))
    x1 = min(bounds[2], max(o.bbox[2] for o in columns))
    y0 = max(bounds[1], min(o.bbox[1] for o in rows))
    y1 = min(bounds[3], max(o.bbox[3] for o in rows))
    xs, ys = _boundaries(columns, x0, x1, 0), _boundaries(rows, y0, y1, 1)
    if any(a >= b for bounds in (xs, ys) for a, b in zip(bounds, bounds[1:], strict=False)):
        table.issues.append(
            Issue(
                code="invalid_axes",
                stage="E",
                severity="error",
                message="Refined axes do not have strictly increasing boundaries",
            )
        )
        return table
    prefix = f"{table_id}-g{table.grid_revision}"
    headers = [o for o in candidates if o.label == "table column header"]
    table.rows = [
        Axis(
            id=f"{prefix}-r{i + 1}",
            index=i + 1,
            bbox=(x0, ys[i], x1, ys[i + 1]),
            coordinate_frame=frame,
            score=obj.score,
            is_header=any(
                intersection_area(obj.bbox, h.bbox) / area(obj.bbox) > 0.5 for h in headers
            ),
        )
        for i, obj in enumerate(rows)
    ]
    table.columns = [
        Axis(
            id=f"{prefix}-c{i + 1}",
            index=i + 1,
            bbox=(xs[i], y0, xs[i + 1], y1),
            coordinate_frame=frame,
            score=obj.score,
        )
        for i, obj in enumerate(columns)
    ]
    positions = {
        (r, c): (xs[c], ys[r], xs[c + 1], ys[r + 1])
        for r in range(len(rows))
        for c in range(len(columns))
    }
    merged: list[tuple[list[int], list[int]]] = []
    occupied: set[tuple[int, int]] = set()
    spans = suppress_duplicates(
        [
            o
            for o in candidates
            if o.label
            in {
                "table spanning cell",
                "table projected row header",
            }
        ]
    )
    for span in spans:
        covered = {
            p for p, box in positions.items() if intersection_area(box, span.bbox) / area(box) > 0.5
        }
        if len(covered) < 2:
            continue
        rs, cs = sorted({p[0] for p in covered}), sorted({p[1] for p in covered})
        rectangle = {(r, c) for r in range(rs[0], rs[-1] + 1) for c in range(cs[0], cs[-1] + 1)}
        if covered != rectangle or occupied & covered:
            table.issues.append(
                Issue(
                    code="conflicting_span",
                    stage="E",
                    severity="error",
                    message="Spanning-cell predictions have conflicting ownership",
                )
            )
            continue
        occupied.update(covered)
        merged.append((rs, cs))
    groups = merged + [([r], [c]) for r, c in positions if (r, c) not in occupied]
    for rs, cs in sorted(groups, key=lambda group: (group[0][0], group[1][0])):
        box = (xs[cs[0]], ys[rs[0]], xs[cs[-1] + 1], ys[rs[-1] + 1])
        table.cells.append(
            Cell(
                id=f"{prefix}-r{rs[0] + 1}-c{cs[0] + 1}",
                row=rs[0] + 1,
                column=cs[0] + 1,
                row_span=len(rs),
                column_span=len(cs),
                table_bbox=box,
                bbox=transform_box(box, table_to_page),
                row_ids=[table.rows[r].id for r in rs],
                column_ids=[table.columns[c].id for c in cs],
                is_header=all(table.rows[r].is_header for r in rs),
            )
        )
    table.geometry_valid = not table.issues
    return table


def annotated_page(image: Image.Image, tables: list[Table], max_pixels: int) -> Image.Image:
    """Draw page-coordinate boxes without cropping, resizing, or adding a legend canvas."""
    if image.width * image.height > max_pixels:
        raise ValueError("Annotated page exceeds configured pixel limit")
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default(size=12)
    for table in tables:
        x0, y0, x1, y1 = table.bbox
        draw.rectangle(table.bbox, outline="green", width=2)
        coords = ",".join(f"{v:.0f}" for v in table.bbox)
        draw.text((x0, max(0, y0 - 32)), f"{table.id} [{coords}]", fill="green", font=font)
        for row in table.rows:
            draw.rectangle(row.bbox, outline="blue", width=1)
            draw.text((max(0, x0 - 48), row.bbox[1]), f"R{row.index}", fill="blue", font=font)
        for column in table.columns:
            draw.rectangle(column.bbox, outline="darkorange", width=1)
            draw.text(
                (column.bbox[0], max(0, y0 - 16)),
                f"C{column.index}",
                fill="darkorange",
                font=font,
            )
    return overlay
