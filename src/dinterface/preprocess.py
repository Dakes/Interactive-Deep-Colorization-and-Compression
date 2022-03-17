"""
Calling this file from commandline, preprocesses all images with default parameters
"""

import os
import cv2
import numpy as np

from src.dinterface.utils import add_color_pixels_rand_gt, imsegkmeans
import src.utils.files as files


def get_fn_wo_ext(fn):
    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        fn = os.path.splitext(fn)[0]
    return fn

def theme_gen_save(dirs, img, filename, num_points_theme=6):
    """
    Generates global theme and saves to disk. (to paths in dir)
    """
    segmented_img, theme, mask = imsegkmeans(img, num_points_theme)
    filename = get_fn_wo_ext(filename)
    cv2.imwrite(dirs["color_map"] + filename + '.png', segmented_img)
    cv2.imwrite(dirs["theme_rgb"] + filename + '.png', theme)
    cv2.imwrite(dirs["theme_mask"] + filename + '.png', mask)

def local_gen_save(dirs, img, filename, num_points_pix):
    """
    only "dumb" generation of points
    """
    points_mask, points_rgb = add_color_pixels_rand_gt(img, num_points_pix)
    filename = get_fn_wo_ext(filename)
    cv2.imwrite(dirs["local_mask"] + filename + '.png', points_mask)
    cv2.imwrite(dirs["local_hints"] + filename + '.png', points_rgb)

def crop(img, random_crop=256):
    h, w, _ = img.shape
    max_sample_h = h - random_crop
    max_sample_w = w - random_crop
    sample_h = np.random.random_integers(0, max_sample_h)
    sample_w = np.random.random_integers(0, max_sample_w)
    return img[sample_h:sample_h + random_crop, sample_w:sample_w + random_crop]


def convert_grayscale(dirs, img):
    # TODO: implement
    pass

# TODO: later (joint training) use same crop for every file (same random seed??)
def preprocess_grayscale(dirs, random_crop=256):
    if not dirs:
        dirs = files.config_parse(dirs=True)
    for dirpath, dnames, fnames in os.walk(dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                temp_fn = get_fn_wo_ext(file)
                item_path = os.path.join(dirpath, file)
                img = cv2.imread(item_path)
                if random_crop:
                    img = crop(img, random_crop)
                # TODO: implement
                img = convert_grayscale(dirs, img)
                cv2.imwrite(dirs["original_img"] + temp_fn + '.png', img)

def preprocess_color(dirs, num_points_pix, num_points_theme, random_crop=256, align=False, only_locals=False):
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

    for dirpath, dnames, fnames in os.walk(dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                temp_fn = get_fn_wo_ext(file)

                item_path = os.path.join(dirpath, file)
                img = cv2.imread(item_path)
                if random_crop:
                    img = crop(dirs, img, random_crop)

                if align:
                    h, w, _ = img.shape
                    if h > w:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                local_gen_save(dirs, img, temp_fn, num_points_pix)
                if not only_locals:
                    theme_gen_save(dirs, img, temp_fn, num_points_theme)
                    cv2.imwrite(dirs["original_img"] + temp_fn + '.png', img)



def main():
    preprocess_color(dirs=None, num_points_pix=-1, num_points_theme=-1, random_crop=False)
    preprocess_grayscale(dirs=None, random_crop=False)


if __name__ == '__main__':
    main()
