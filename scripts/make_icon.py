"""生成 uvr-lite ♪ 图标（ico + png）。

用法: python scripts/make_icon.py
输出: uvr_lite/ui/resources/uvr-lite.ico（多尺寸）、uvr-lite.png（256px）
字体: 依次尝试 Segoe UI Symbol / Segoe UI Emoji / DejaVu Sans，均无则用默认字体。
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "uvr_lite" / "ui" / "resources"
SIZE = 256
BG = (32, 36, 48, 255)      # 深蓝灰圆角底
FG = (255, 255, 255, 255)   # 白色音符
RADIUS = 56                 # 圆角半径

FONT_CANDIDATES = [
    ("C:/Windows/Fonts/seguisym.ttf", 190),   # Segoe UI Symbol（含 U+266A ♪）
    ("C:/Windows/Fonts/seguiemj.ttf", 190),   # Segoe UI Emoji（回退）
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 190),  # Linux
    ("C:/Windows/Fonts/arial.ttf", 190),      # 最后回退
]


def find_font() -> ImageFont.FreeTypeFont:
    for path, size in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def draw_icon() -> Image.Image:
    """深色圆角底 + 白色 ♪，256x256 RGBA。"""
    img = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, SIZE - 1, SIZE - 1], radius=RADIUS, fill=BG)

    font = find_font()
    glyph = "\u266a"  # ♪
    bbox = d.textbbox((0, 0), glyph, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # 水平居中；垂直方向微调 8px（♪ 视觉重心略偏下）
    x = (SIZE - w) / 2 - bbox[0]
    y = (SIZE - h) / 2 - bbox[1] - 8
    d.text((x, y), glyph, font=font, fill=FG)
    return img


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img = draw_icon()
    img.save(OUT_DIR / "uvr-lite.png")
    img.save(OUT_DIR / "uvr-lite.ico", sizes=[
        (16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256),
    ])
    print(f"OK: {OUT_DIR / 'uvr-lite.png'}")
    print(f"OK: {OUT_DIR / 'uvr-lite.ico'}")


if __name__ == "__main__":
    main()
