import os
import random
from os.path import join
import PIL
from PIL import Image
import skimage.color
import numpy as np
import argparse
import warnings


NUM_TASKS = 1
job_enumerate = enumerate

warnings.filterwarnings("ignore")


QUALITY = 95
MAX_SCALE = 0.95


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base_dir")
    p.add_argument("dirs", nargs="+")
    p.add_argument("--out_dir_clean", required=True)
    p.add_argument("--out_dir_discard", required=True)

    p.add_argument("--resolution", "-r", type=str, default="768", help="can be randX_Y")
    p.add_argument("--seed")
    flags = p.parse_args()

    if flags.seed:
        print("Seeding {}".format(flags.seed))
        random.seed(flags.seed)

    for d in flags.dirs:
        process(
            join(flags.base_dir, d),
            flags.out_dir_clean,
            flags.out_dir_discard,
            flags.resolution,
        )


def get_res(res) -> int:
    try:
        return int(res)
    except ValueError:
        pass
    if not res.startswith("rand"):
        raise ValueError("Expected res to be either int or `randX_Y`")
    X, Y = map(int, res.replace("rand", "").split("_"))
    return random.randint(X, Y)


def process(input_dir, out_dir_clean, out_dir_discard, res: str):
    os.makedirs(out_dir_clean, exist_ok=True)
    os.makedirs(out_dir_discard, exist_ok=True)

    images_cleaned = set(os.listdir(out_dir_clean))
    images_discarded = set(os.listdir(out_dir_discard))

    images_dl = os.listdir(input_dir)
    N = len(images_dl) // NUM_TASKS

    clean = 0
    discarded = 0
    for i, imfile in job_enumerate(images_dl):
        if imfile in images_cleaned:
            clean += 1
            continue
        if imfile in images_discarded:
            discarded += 1
            continue
        im = Image.open(join(input_dir, imfile))
        res = get_res(res)
        im2 = resize_or_discard(im, res, should_clean=True)
        if im2 is not None:
            fn, ext = os.path.splitext(imfile)
            im2.save(join(out_dir_clean, fn + ".png"))
            clean += 1
        else:
            im.save(join(out_dir_discard, imfile))
            discarded += 1
        print(
            f"\r{os.path.basename(input_dir)} -> {os.path.basename(out_dir_clean)} // "
            f"Resized: {clean}/{N}; Discarded: {discarded}/{N}",
            end="",
        )
    # Done
    print(
        f"\n{os.path.basename(input_dir)} -> {os.path.basename(out_dir_clean)} // "
        f"Resized: {clean}/{N}; Discarded: {discarded}/{N}"
    )


def resize_or_discard(im, res: int, verbose=False, should_clean=True):
    im2 = resize(im, res, verbose)
    if im2 is None:
        return None
    if should_clean and should_discard(im2, verbose):
        return None
    return im2


def resize(im, res, verbose=False, max_scale=MAX_SCALE):
    W, H = im.size
    D = max(W, H)
    s = float(res) / D
    if max_scale and s > max_scale:
        if verbose:
            print("Too big: {}".format((W, H)))
        return None
    W2 = round(W * s)
    H2 = round(H * s)
    try:
        return im.resize((W2, H2), resample=PIL.Image.BICUBIC)
    except OSError as e:
        print(e)
        return None


def should_discard(im, verbose=False):
    im_rgb = np.array(im)
    if im_rgb.ndim != 3 or im_rgb.shape[2] != 3:
        if verbose:
            print("Invalid shape: {}".format(im_rgb.shape))
        return True
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    h, s, v = mean_hsv
    if s > 0.9:
        if verbose:
            print("Invalid s: {}".format(s))
        return True
    if v > 0.8:
        if verbose:
            print("Invalid v: {}".format(v))
        return True
    return False


def get_hsv(im):
    im_rgb = np.array(im)
    im_hsv = skimage.color.rgb2hsv(im_rgb)
    mean_hsv = np.mean(im_hsv, axis=(0, 1))
    h, s, v = mean_hsv
    return h, s, v


if __name__ == "__main__":
    main()
