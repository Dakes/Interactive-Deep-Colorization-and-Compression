"""
Calling this file from commandline, preprocesses all images with default parameters
"""

import os, sys
import cv2
import numpy as np

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.dinterface.utils import add_color_pixels_rand_gt, imsegkmeans
# from src.dinterface.utils import add_color_pixels_rand_gt, imsegkmeans
from src.utils import files
# import src.utils.files as files


def get_fn_wo_ext(fn):
    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        fn = os.path.splitext(fn)[0]
    return fn

def theme_gen_save(dirs, img, filename, num_points_theme=6, save_segmented=False):
    """
    Generates global theme and saves to disk. (to paths in dir)
    """
    segmented_img, theme, mask = imsegkmeans(img, num_points_theme)
    filename = get_fn_wo_ext(filename)
    if save_segmented:
        cv2.imwrite(dirs["color_map"] + filename + '.png', segmented_img)
    cv2.imwrite(dirs["theme_rgb"] + filename + '.png', theme, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
    cv2.imwrite(dirs["theme_mask"] + filename + '.png', mask, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

def local_gen_save(dirs, img, filename, num_points_pix):
    """
    only "dumb" generation of points
    """
    points_mask, points_rgb = add_color_pixels_rand_gt(img, num_points_pix)
    filename = get_fn_wo_ext(filename)
    cv2.imwrite(dirs["local_mask"] + filename + '.png', points_mask, [int(cv2.IMWRITE_PNG_COMPRESSION),0])
    cv2.imwrite(dirs["local_hints"] + filename + '.png', points_rgb, [int(cv2.IMWRITE_PNG_COMPRESSION),0])

def crop(img, random_crop=256):
    # TODO: save crops (needed for dynamic point selection). (Except seed is used [derived from filename?])
    h, w, _ = img.shape
    max_sample_h = h - random_crop
    max_sample_w = w - random_crop
    sample_h = np.random.random_integers(0, max_sample_h)
    sample_w = np.random.random_integers(0, max_sample_w)
    return img[sample_h:sample_h + random_crop, sample_w:sample_w + random_crop]


# TODO: later (joint training) use same crop for every file (same random seed??)
def preprocess_grayscale(dirs, random_crop=256):
    # NOTE: Naaah, thats stupid, convert to grayscale on demand. Is quicker than saving and loading twice
    # (Unless we use a ramdisk to save the 140gb dataset, which we can totally do *flex*)
    if not dirs:
        dirs = files.config_parse(dirs=True)

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

def preprocess_color(dirs, num_points_pix, num_points_theme, random_crop=256, align=False, only_locals=False,
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
    for dirpath, dnames, fnames in os.walk(dirs):
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
                    try:
                        img = crop(img, random_crop)
                    except ValueError:
                        continue

                if align:
                    h, w, _ = img.shape
                    if h > w:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                # local_gen_save(dirs, img, temp_fn, num_points_pix)
                if not only_locals:
                    # theme_gen_save(dirs, img, temp_fn, num_points_theme, save_segmented)
                    # save alined?? But not like this
                    cv2.imwrite("/home/daniel/PycharmProjects/Interactive-Deep-Colorization-and-Compression/res/eval/original_img/" + '/' + temp_fn + '.png', img, [int(cv2.IMWRITE_PNG_COMPRESSION), 0]) #TODO adjust path


def dataset_prepare(dirs):
    """
    Links dataset into 'original_img'. Runs recursively.
    ALL images below given folder will be used.
    REMOVES all old links
    """
    # remove all old links
    print("Deleting old symlinks in:", dirs["original_img"])
    for dirpath, _, fnames in os.walk(dirs["original_img"]):
        for file in fnames:
            link = os.path.join(dirpath, file)
            if os.path.islink(link):
                os.unlink(link)

    counter = 0
    for dirpath, _, fnames in os.walk(dirs["dataset"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                counter += 1
                if not counter % 10000:
                    print("dataset_prepare; Symlinked", counter, "images")
                src_file = os.path.join(dirpath, file)
                new_link = os.path.join(dirs["original_img"], file)
                os.symlink(src_file, new_link)

def main():
    dirs = files.config_parse(dirs=True)
    dataset_prepare(dirs)
    preprocess_color(dirs, num_points_pix=-1, num_points_theme=-1, random_crop=False)
    preprocess_grayscale(dirs, random_crop=False)


if __name__ == '__main__':
    main()
