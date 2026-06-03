import cv2
import glob
import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

GOOD_PATH = 'dataset/valid/'
BAD_PATH  = 'dataset/invalid/'

OUT_GOOD  = 'dataset_aug/valid/'
OUT_BAD   = 'dataset_aug/invalid/'

MODEL_SIZE    = (80, 80)
WORKING_SIZE  = (160, 160)  # upscale before foreground extraction


def extract_foreground(img):
    # your existing function
    ...

#Find a good library for augmentation!
def augment_image(img: Image.Image) -> list[Image.Image]:
    variants = []
    variants.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    variants.append(img.transpose(Image.FLIP_TOP_BOTTOM))
    variants.append(img.rotate(90))
    variants.append(ImageEnhance.Brightness(img).enhance(0.8))
    variants.append(ImageEnhance.Contrast(img).enhance(1.3))
    variants.append(img.filter(ImageFilter.GaussianBlur(radius=1)))
    return variants


def process_image(path: str) -> Image.Image | None:
    """Load, upscale, extract foreground, downscale to model size."""
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: could not read {path}")
        return None

    # Upscale for better foreground extraction
    img = cv2.resize(img, WORKING_SIZE, interpolation=cv2.INTER_LINEAR)

    # Extract foreground (on the larger image)
    fg = extract_foreground(img)

    # Downscale to model input size
    fg = cv2.resize(fg, MODEL_SIZE, interpolation=cv2.INTER_AREA)

    # Convert to PIL
    return Image.fromarray(cv2.cvtColor(fg, cv2.COLOR_BGR2RGB))


def save_image(img: Image.Image, out_dir: str, base: str, suffix: str):
    os.makedirs(out_dir, exist_ok=True)
    img.save(os.path.join(out_dir, f'{base}_{suffix}.jpg'))


def prepare_dataset():
    good_paths = glob.glob(os.path.join(GOOD_PATH, '*.jpg'))
    good_paths += glob.glob(os.path.join(GOOD_PATH, '*.png'))
    bad_paths  = glob.glob(os.path.join(BAD_PATH,  '*.jpg'))
    bad_paths += glob.glob(os.path.join(BAD_PATH,  '*.png'))

    print(f"Found {len(good_paths)} good, {len(bad_paths)} bad images")

    # --- Good images: process + augment ---
    for path in good_paths:
        pil  = process_image(path)
        if pil is None:
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(pil, OUT_GOOD, base, 'orig')
        for i, aug in enumerate(augment_image(pil)):
            save_image(aug, OUT_GOOD, base, f'aug{i}')

    # --- Bad images: process only, no augmentation ---
    for path in bad_paths:
        pil  = process_image(path)
        if pil is None:
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        save_image(pil, OUT_BAD, base, 'orig')

    good_out = len(glob.glob(OUT_GOOD + '*.jpg'))
    bad_out  = len(glob.glob(OUT_BAD  + '*.jpg'))
    print(f"Output — Good: {good_out}, Bad: {bad_out}")

#Fix bad paths to be only one, since no augmentation is done there so we only need to extract the foregrounds and pass that.

if __name__ == '__main__':
    prepare_dataset()

#Separate this part
WORKING_SIZE = (160, 160)
MODEL_SIZE   = (80, 80)

def detect_olive_fly(image) -> bool:
    if isinstance(image, str):
        img = cv2.imread(image)
    else:
        img = image

    # Identical preprocessing to training
    img = cv2.resize(img, WORKING_SIZE, interpolation=cv2.INTER_LINEAR)
    fg  = extract_foreground(img)
    fg  = cv2.resize(fg, MODEL_SIZE, interpolation=cv2.INTER_AREA)

    arr = cv2.cvtColor(fg, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)

    _interpreter.set_tensor(_input_details[0]['index'], arr)
    _interpreter.invoke()
    score = _interpreter.get_tensor(_output_details[0]['index'])[0][0]

    return bool(score >= _THRESHOLD)