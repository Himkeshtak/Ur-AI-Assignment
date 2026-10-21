"""Generate the synthetic invoice used for manual end-to-end checks (no real data)."""

import argparse
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def create_demo(path: Path, font_path: Path) -> None:
    pdfmetrics.registerFont(TTFont("DemoJapanese", str(font_path), subfontIndex=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf = canvas.Canvas(str(path), pagesize=(600, 700), invariant=True)
    pdf.setTitle("Synthetic Japanese invoice - parser integration check")
    pdf.setFont("DemoJapanese", 19)
    pdf.drawString(50, 630, "請求書（テスト用・架空の取引）")
    pdf.setFont("DemoJapanese", 10)
    pdf.drawString(50, 595, "日本語・桁区切り・行の対応を確認するための合成データです。")
    rows = [
        ("品目", "数量", "単価", "金額"),
        ("用紙", "2", "1,000", "2,000"),
        ("封筒", "1", "300", "300"),
        ("小計", "", "", "2,300"),
        ("消費税", "", "", "230"),
        ("合計", "", "", "2,530"),
    ]
    boundaries = [50, 235, 315, 425, 550]
    top, height = 550, 48
    for r, row in enumerate(rows):
        bottom = top - (r + 1) * height
        for c, text in enumerate(row):
            pdf.rect(boundaries[c], bottom, boundaries[c + 1] - boundaries[c], height)
            pdf.setFont("DemoJapanese", 13)
            pdf.drawString(boundaries[c] + 12, bottom + 18, text)
    pdf.setFont("DemoJapanese", 10)
    pdf.drawString(50, 220, "消費税率 10% / 外税 / 金額は円単位")
    pdf.showPage()
    pdf.save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--font", type=Path, required=True, help="Japanese TrueType font/collection"
    )
    parser.add_argument("--output", type=Path, default=Path("output/pdf/synthetic-invoice.pdf"))
    args = parser.parse_args()
    create_demo(args.output, args.font)
    print(args.output)
