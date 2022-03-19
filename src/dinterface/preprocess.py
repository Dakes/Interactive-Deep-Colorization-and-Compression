"""
Calling this file from commandline, preprocesses all images with default parameters
"""

import os, sys
import cv2
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from utils import add_color_pixels_rand_gt, imsegkmeans
from src.utils import files

dirs = files.config_parse(dirs=True)

def get_fn_wo_ext(fn):
    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        fn = os.path.splitext(fn)[0]
    return fn

def cache_img(img, filepath, png_compression=0):
    """
    Saves to disk, only if not already present
    """
    if not os.path.isfile(filepath):
        cv2.imwrite(filepath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    else:
        # TODO: maybe load and compare to new image & save if changed
        pass

def load_cache_img(filepath):
    """
    :return: None if file not exists
    """
    img = None
    if os.path.isfile(filepath):
        img = cv2.imread(filepath)
    return img

def theme_gen(img, filename, set="train", num_points_theme=6, save_segmented=False, overwrite=False):
    """
    :param set: train, val (or test)
    Generates global theme. Caches to disk, if not already cached.
    Does nothing if files exist on disk.
    :param overwrite: if True, overwrite existing files
    """
    filename = get_fn_wo_ext(filename)
    theme_path = dirs[set] + dirs["theme_rgb"] + filename + '.png'
    mask_path = dirs[set] + dirs["theme_mask"] + filename + '.png'
    if overwrite or not os.path.isfile(theme_path) or not os.path.isfile(mask_path):
        segmented_img, theme, mask = imsegkmeans(img, num_points_theme)
        cache_img(theme, theme_path)
        cache_img(mask, mask_path)
        if save_segmented:
            segmented_path = dirs[set] + dirs["color_map"] + filename + '.png'
            if not os.path.isfile(segmented_path):
                cache_img(segmented_img, segmented_path)


def local_gen(img, filename, num_points_pix=-1, set="train", overwrite=False):
    """
    Generates local points. Caches if not already on disk. Skips, if present.
    only "dumb" generation of points
    :param num_points_pix: -1; random number
    :param overwrite: if True, overwrite existing files
    """
    filename = get_fn_wo_ext(filename)
    mask_path = dirs[set] + dirs["local_mask"] + filename + '.png'
    points_path = dirs[set] + dirs["local_hints"] + filename + '.png'

    if overwrite or not os.path.isfile(mask_path) or not os.path.isfile(points_path):
        points_mask, points_rgb = add_color_pixels_rand_gt(img, num_points_pix)
        cache_img(points_mask, mask_path)
        cache_img(points_rgb, points_path)

def crop(img, random_crop=256):
    # TODO: save crops (needed for dynamic point selection). (Except seed is used [derived from filename?])
    h, w, _ = img.shape
    max_sample_h = h - random_crop
    max_sample_w = w - random_crop
    sample_h = np.random.random_integers(0, max_sample_h)
    sample_w = np.random.random_integers(0, max_sample_w)
    return img[sample_h:sample_h + random_crop, sample_w:sample_w + random_crop]


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
                if random_crop:
                    img = crop(img, random_crop)

                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
    if not dirs:
        dirs = files.config_parse(dirs=True)

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
                if not counter % 10000:
                    print("dataset_prepare; Symlinked", counter, "images")
                src_file = os.path.join(dirpath, file)
                new_link = os.path.join(dirs[set] + dirs["original_img"], file)
                os.symlink(src_file, new_link)

def main():
    dataset_prepare("train")
    dataset_prepare("val")
    preprocess_color(num_points_pix=-1, num_points_theme=-1, random_crop=False)
    preprocess_grayscale(random_crop=False)


if __name__ == '__main__':
    main()
