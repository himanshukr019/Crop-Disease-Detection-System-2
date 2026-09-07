"""
generate_dataset.py
-------------------
Generates synthetic leaf images for training/testing the CNN model.
Run this FIRST before train.py if you don't have a real dataset.

Usage: python generate_dataset.py
"""

import os
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────
CLASSES = {
    "Healthy":          {"base_color": (34, 139, 34),  "spot_color": None,           "n_train": 80, "n_test": 20},
    "Powdery_Mildew":   {"base_color": (60, 120, 40),  "spot_color": (230, 230, 220),"n_train": 80, "n_test": 20},
    "Leaf_Blight":      {"base_color": (50, 110, 30),  "spot_color": (160, 100, 40), "n_train": 80, "n_test": 20},
    "Rust":             {"base_color": (45, 100, 35),  "spot_color": (180, 80,  20), "n_train": 80, "n_test": 20},
    "Bacterial_Spot":   {"base_color": (40, 115, 38),  "spot_color": (80,  60,  30), "n_train": 80, "n_test": 20},
}

IMG_SIZE   = 128
OUTPUT_DIR = "dataset"
# ──────────────────────────────────────────────────────────────────────────────


def draw_leaf_shape(draw, size, color):
    """Draw a stylised elliptical leaf with slight noise on the boundary."""
    cx, cy = size // 2, size // 2
    rx, ry = int(size * 0.38), int(size * 0.46)
    points = []
    for deg in range(0, 360, 3):
        rad = math.radians(deg)
        noise = random.uniform(0.88, 1.0)
        x = cx + int(rx * math.cos(rad) * noise)
        y = cy + int(ry * math.sin(rad) * noise)
        points.append((x, y))
    r, g, b = color
    draw.polygon(points, fill=(r, g, b, 220))


def add_veins(draw, size, color):
    """Draw simple leaf veins."""
    cx, cy = size // 2, size // 2
    r, g, b = color
    vein_color = (max(r - 20, 0), max(g - 25, 0), max(b - 10, 0), 180)
    # main midrib
    draw.line([(cx, cy - int(size * 0.42)), (cx, cy + int(size * 0.42))],
              fill=vein_color, width=2)
    # lateral veins
    for sign in [-1, 1]:
        for i in range(1, 5):
            y_start = cy + sign * i * int(size * 0.08)
            x_end   = cx + int(size * 0.30) * (1 if sign == 1 else -1)
            draw.line([(cx, y_start), (x_end, y_start - int(size * 0.06))],
                      fill=vein_color, width=1)


def add_spots(draw, size, spot_color, disease_type, count=None):
    """Add disease-specific spots/patches."""
    if spot_color is None:
        return

    r, g, b = spot_color
    cx, cy  = size // 2, size // 2
    count   = count or random.randint(6, 18)

    for _ in range(count):
        # Keep spots mostly inside the leaf ellipse
        angle  = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, size * 0.30)
        sx = int(cx + radius * math.cos(angle))
        sy = int(cy + radius * math.sin(angle) * 1.2)

        if disease_type == "Powdery_Mildew":
            # Irregular white-ish powder patches
            sr = random.randint(6, 14)
            for dx in range(-sr, sr + 1, 2):
                for dy in range(-sr, sr + 1, 2):
                    if dx*dx + dy*dy <= sr*sr:
                        alpha = random.randint(100, 200)
                        draw.point((sx + dx, sy + dy), fill=(r, g, b, alpha))

        elif disease_type == "Leaf_Blight":
            # Large irregular brown patches
            sr = random.randint(8, 18)
            pts = [(sx + int(sr * math.cos(math.radians(a)) * random.uniform(0.7, 1.3)),
                    sy + int(sr * math.sin(math.radians(a)) * random.uniform(0.7, 1.3)))
                   for a in range(0, 360, 20)]
            draw.polygon(pts, fill=(r, g, b, 200))

        elif disease_type == "Rust":
            # Small orange-brown circular pustules
            sr = random.randint(3, 7)
            draw.ellipse([(sx - sr, sy - sr), (sx + sr, sy + sr)],
                         fill=(r, g, b, 220))
            # Darker centre
            cr = max(1, sr - 2)
            draw.ellipse([(sx - cr, sy - cr), (sx + cr, sy + cr)],
                         fill=(max(r-30,0), max(g-20,0), max(b-10,0), 240))

        elif disease_type == "Bacterial_Spot":
            # Small dark angular spots
            sr = random.randint(3, 8)
            pts = [(sx + int(sr * math.cos(math.radians(a))),
                    sy + int(sr * math.sin(math.radians(a))))
                   for a in range(0, 360, 45)]
            draw.polygon(pts, fill=(r, g, b, 220))


def generate_leaf_image(class_name, cfg):
    """Create one synthetic leaf image."""
    img  = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (200, 220, 200, 0))
    draw = ImageDraw.Draw(img)

    # Background gradient (sky light)
    bg = Image.new("RGBA", (IMG_SIZE, IMG_SIZE), (210, 230, 200, 255))
    img = Image.alpha_composite(bg, img)
    draw = ImageDraw.Draw(img)

    # Leaf body
    draw_leaf_shape(draw, IMG_SIZE, cfg["base_color"])
    add_veins(draw, IMG_SIZE, cfg["base_color"])

    if cfg["spot_color"]:
        add_spots(draw, IMG_SIZE, cfg["spot_color"], class_name)

    # Slight blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    # Add mild noise
    np_img = np.array(img.convert("RGB")).astype(np.float32)
    noise  = np.random.normal(0, 6, np_img.shape)
    np_img = np.clip(np_img + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(np_img)


def main():
    print("🌿 Generating synthetic leaf dataset …")
    total = 0
    for cls, cfg in CLASSES.items():
        for split, n in [("train", cfg["n_train"]), ("test", cfg["n_test"])]:
            out_dir = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(out_dir, exist_ok=True)
            for i in range(n):
                img  = generate_leaf_image(cls, cfg)
                path = os.path.join(out_dir, f"{cls}_{split}_{i:04d}.jpg")
                img.save(path, "JPEG", quality=90)
                total += 1
        print(f"  ✅  {cls}: {cfg['n_train']} train  /  {cfg['n_test']} test")

    print(f"\n✨ Done! {total} images saved to ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
