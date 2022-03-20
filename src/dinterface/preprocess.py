"""
Calling this file from commandline, preprocesses all images with default parameters
"""

import os, sys
import cv2
import numpy as np
import pathlib

sys.path.insert(1, os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../..')))
sys.path.insert(1, os.path.abspath(pathlib.Path(__file__).parent.resolve()))
from dutils import add_color_pixels_rand_gt, imsegkmeans, arr2tf, get_fn_wo_ext
from src.utils import files

dirs = files.config_parse(dirs=True)


def cache_img(img, filepath, overwrite, png_compression=0):
    """
    Saves to disk, only if not already present
    """
    try:
        if not os.path.isfile(filepath):
            cv2.imwrite(filepath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
        elif overwrite:
            # TODO: load and compare to new image & save if changed
            cv2.imwrite(filepath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
        else:
            pass
    except FileNotFoundError as err:
        print("Creating dir:", os.path.dirname(filepath))
        os.makedirs(os.path.dirname(filepath))
        cache_img(img, filepath, overwrite, png_compression)


def load_img(filepath):
    """
    :return: None if file not exists
    """
    img = None
    if os.path.isfile(filepath):
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def theme_gen(filename, set="train", num_points_theme=6, save_segmented=False, overwrite=False):
    """
    :param img: color image, already cropped, if crop is desired
    :param set: train, val (or test)
    Generates global theme. Caches to disk, if not already cached.
    Does nothing if files exist on disk.
    :param overwrite: if True, overwrite existing files
    """
    filename = get_fn_wo_ext(filename)
    theme_path = dirs[set] + dirs["theme_rgb"] + filename + '.png'
    mask_path = dirs[set] + dirs["theme_mask"] + filename + '.png'
    segmented_path = dirs[set] + dirs["color_map"] + filename + '.png'
    theme = load_img(theme_path)
    mask = load_img(mask_path)
    segmented_img = load_img(segmented_path)

    if overwrite or not theme or not mask:
        gt_path = dirs[set] + dirs["ground_truth"] + filename + ".png"
        gt = load_img(gt_path)
        segmented_img, theme, mask = imsegkmeans(gt, num_points_theme)
        cache_img(theme, theme_path, overwrite)
        cache_img(mask, mask_path, overwrite)
        if save_segmented:
            cache_img(segmented_img, segmented_path)
    return arr2tf(theme), arr2tf(mask), arr2tf(segmented_img)


def local_gen(filename, num_points_pix=-1, set="train", overwrite=False):
    """
    Generates local points. Caches if not already on disk. Skips, if present.
    only "dumb" generation of points
    :param img: color image, already cropped, if crop is desired
    :param num_points_pix: -1; random number
    :param overwrite: if True, overwrite existing files
    """
    filename = get_fn_wo_ext(filename)
    mask_path = dirs[set] + dirs["local_mask"] + filename + '.png'
    points_path = dirs[set] + dirs["local_hints"] + filename + '.png'
    points_rgb = load_img(points_path)
    points_mask = load_img(mask_path)

    if overwrite or not points_rgb or not points_mask:
        gt_path = dirs[set] + dirs["ground_truth"] + filename + ".png"
        gt = load_img(gt_path)
        points_mask, points_rgb = add_color_pixels_rand_gt(gt, num_points_pix)
        cache_img(points_mask, mask_path, overwrite)
        cache_img(points_rgb, points_path, overwrite)
    return arr2tf(points_rgb), arr2tf(points_mask)


def crop(img, random_crop=256, fn=""):
    """
    :param fn: will be incorporated into random seed (same crop for gray & color) (w/o extension)
    :return: cropped img
    """
    if fn:
        hsh = abs(hash(get_fn_wo_ext(fn))) + random_crop
    else:
        hsh = np.random.randint(9999999)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(hsh)))

    h, w, _ = img.shape
    max_sample_h = h - random_crop
    max_sample_w = w - random_crop
    sample_h = rs.randint(0, max_sample_h)
    sample_w = rs.randint(0, max_sample_w)
    return img[sample_h:sample_h + random_crop, sample_w:sample_w + random_crop]

def _gt_gen(img, filename, set="train", random_crop=256, overwrite=False):
    filename = get_fn_wo_ext(filename)
    gt_path = dirs[set] + dirs["ground_truth"] + filename + ".png"

    if overwrite or not os.path.isfile(gt_path):
        img = crop(img, random_crop, filename)
        cache_img(img, gt_path, overwrite)

def gt_gen_color(filename, set="train", random_crop=256, overwrite=False):
    """
    :param filename: original img path
    """
    img = load_img(filename)
    return arr2tf(_gt_gen(img, filename, set=set, random_crop=random_crop, overwrite=overwrite))

def gt_gen_gray(filename, set="train", random_crop=256, overwrite=False):
    """
    :param filename: original img path
    """
    img = load_img(filename)
    return arr2tf(_gt_gen(img, filename, set=set, random_crop=random_crop, overwrite=overwrite))


# TODO: later (joint training) use same crop for every file (same random seed??)
# LEGACY FUNCTION
def preprocess_grayscale(random_crop=256):
    counter = 0
    for dirpath, dnames, fnames in os.walk(dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                counter += 1
                if not counter % 10000:
                    print("preprocess_grayscale; Processed", counter, "images")
                temp_fn = get_fn_wo_ext(file)
                item_path = os.path.join(dirpath, file)
                img = cv2.imread(item_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if random_crop:
                    img = crop(img, random_crop)

                # cv2.imwrite(dirs["original_gray_img"] + temp_fn + '.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

# LEGACY FUNCTION
def preprocess_color(num_points_pix, num_points_theme, random_crop=256, align=False, only_locals=False,
                     save_segmented=False):
    """
    clean python implementation of:
    https://github.com/praywj/Interactive-Deep-Colorization-and-Compression/tree/master/prepare_dataset
    iterates through "original_img" folder, to prepare color cues

    :param dirs: config parser (dict-like) with directories
    :param num_points_pix:
    :param num_points_theme:
    :param random_crop: target size of crop. If false/0, don't crop
    :param align:
    :param only_locals:
    :return:
    """
    counter = 0
    for dirpath, dnames, fnames in os.walk(dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                counter += 1
                print(file)
                if not counter % 1000:
                    print("preprocess_color; Processed", counter, "images")
                temp_fn = get_fn_wo_ext(file)

                item_path = os.path.join(dirpath, file)
                img = cv2.imread(item_path)
                if random_crop:
                    img = crop(dirs, img, random_crop)

                if align:
                    h, w, _ = img.shape
                    if h > w:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                local_gen(dirs, img, temp_fn, num_points_pix)
                if not only_locals:
                    theme_gen(dirs, img, temp_fn, num_points_theme, save_segmented)
                    # save alined?? But not like this
                    # cv2.imwrite(dirs["original_img"] + temp_fn + '.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION),0])


def dataset_prepare(set="train"):
    """
    :param set: train, val (or test)
    Links dataset into 'original_img'. Runs recursively.
    ALL images below given folder will be used.
    REMOVES all old links
    """
    # to run only once
    try:
        if dataset_prepare.prepared[set]:
            return
    except (AttributeError, KeyError):
        dataset_prepare.prepared = {}
    # remove all old links
    print("Deleting old symlinks in:", dirs[set] + dirs["original_img"])
    for dirpath, _, fnames in os.walk(dirs[set] + dirs["original_img"]):
        for file in fnames:
            link = os.path.join(dirpath, file)
            if os.path.islink(link):
                os.unlink(link)

    counter = 0
    for dirpath, _, fnames in os.walk(dirs["dataset"]+set):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                counter += 1
                if not counter % 100000:
                    print("dataset_prepare; Symlinked", counter, "images for set", set)
                src_file = os.path.join(dirpath, file)
                new_link = os.path.join(dirs[set] + dirs["original_img"], file)
                try:
                    os.symlink(src_file, new_link)
                except FileNotFoundError as err:
                    print("Creating dir:", os.path.dirname(new_link))
                    os.makedirs(os.path.dirname(new_link))
                    os.symlink(src_file, new_link)
    print("dataset_prepare; Symlinked", counter, "images for set", set)
    dataset_prepare.prepared[set] = True

def main():
    dataset_prepare(set="train")
    dataset_prepare(set="val")
    # preprocess_color(num_points_pix=-1, num_points_theme=-1, random_crop=False)
    # preprocess_grayscale(random_crop=False)


if __name__ == '__main__':
    main()
