import os
from pathlib import Path
from PIL import Image, ImageFilter
import torch
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import argparse

from tqdm import tqdm


def compute_antialiasing_sigma(orig_size, target_size):
    scale = max(orig_size[0] / target_size[0], orig_size[1] / target_size[1])
    return 0.5 * scale if scale > 1.0 else 0.0

def process_and_save_image(img_path, out_path, target_size):
    try:
        img = Image.open(img_path).convert("L")  # convert to grayscale

        # Rotate landscape images to portrait
        if img.width > img.height:
            img = img.rotate(90, expand=True)

        sigma = compute_antialiasing_sigma(img.size, target_size)
        if sigma > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=sigma))

        img = img.resize(target_size, resample=Image.Resampling.BILINEAR)

        tensor = transforms.ToTensor()(img)        # shape [1, H, W], in [0, 1]
        tensor = tensor * 2.0 - 1.0                 # rescale to [-1, 1]

        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, out_path)

        return f"✓"
    except Exception as e:
        return f"✗"


def preprocess_all_images(
    input_root="validated_data",
    output_root="down_data",
    target_size=(32, 32),
    num_workers=8,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    tasks = []

    for class_dir in input_root.iterdir():
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
                continue
            rel_path = img_path.relative_to(input_root)
            out_path = output_root / rel_path.with_suffix(".pt")
            tasks.append((img_path, out_path))

    print(f"Processing {len(tasks)} images with {num_workers} workers...")

    # with ProcessPoolExecutor(max_workers=num_workers) as executor:
    func = partial(process_and_save_image, target_size=target_size)
    # add current name of file to the progress bar, which we will have access to in the loop
    bar = tqdm(total=len(tasks), desc="Processing images")

    for task in tasks:
        file = func(*task)
        bar.update(1)
        bar.set_postfix_str(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_size", type=int, nargs=2, default=[64, 32])
    parser.add_argument("--input_root", type=str, default="validated_data")
    parser.add_argument("--output_root", type=str, default="down_data")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    preprocess_all_images(
        input_root=args.input_root,
        output_root=args.output_root,
        target_size=tuple(args.target_size),
        num_workers=args.workers,
    )
