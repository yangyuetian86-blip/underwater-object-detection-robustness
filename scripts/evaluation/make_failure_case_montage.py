from pathlib import Path
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(r"F:/Program Files/Underwater_Detection")

INPUT_DIR = PROJECT_ROOT / "paper_figures" / "selected_failure_cases"
OUTPUT_DIR = PROJECT_ROOT / "paper_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CASE_STEMS = [
    "case_a_holothurian",
    "case_b_scallop",
    "case_c_haze",
    "case_d_severe",
]


def find_image_by_stem(input_dir, stem):
    for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"]:
        path = input_dir / f"{stem}{ext}"
        if path.exists():
            return path

    candidates = list(input_dir.glob(stem + ".*"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"Cannot find image for stem: {stem}")


def resize_and_center_crop(img, target_w, target_h):
    """
    Fill the target cell without white padding.
    The image is center-cropped to the target ratio and then resized.
    """
    img = img.convert("RGB")
    w, h = img.size

    target_ratio = target_w / target_h
    img_ratio = w / h

    if img_ratio > target_ratio:
        new_w = int(h * target_ratio)
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))
    else:
        new_h = int(w / target_ratio)
        top = (h - new_h) // 2
        img = img.crop((0, top, w, top + new_h))

    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return img


def main():
    # 这里把高度加大，避免 A 图左右留白太多
    cell_w = 760
    cell_h = 520

    margin = 10

    total_w = cell_w * 2 + margin * 3
    total_h = cell_h * 2 + margin * 3

    montage = Image.new("RGB", (total_w, total_h), "white")
    draw = ImageDraw.Draw(montage)

    for idx, stem in enumerate(CASE_STEMS):
        img_path = find_image_by_stem(INPUT_DIR, stem)
        print(f"Loading: {img_path}")

        img = Image.open(img_path)
        img = resize_and_center_crop(img, cell_w, cell_h)

        row = idx // 2
        col = idx % 2

        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin)

        montage.paste(img, (x, y))

        draw.rectangle(
            [x, y, x + cell_w - 1, y + cell_h - 1],
            outline="black",
            width=1
        )

    out_png = OUTPUT_DIR / "failure_case_montage.png"
    out_pdf = OUTPUT_DIR / "failure_case_montage.pdf"

    montage.save(out_png, dpi=(300, 300))
    montage.save(out_pdf)

    print("Saved:")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()