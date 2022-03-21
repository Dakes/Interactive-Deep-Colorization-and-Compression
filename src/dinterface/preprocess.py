"""
Calling this file from commandline, preprocesses all images with default parameters
"""

import os, sys
import cv2
import numpy as np
import pathlib
import hashlib
from multiprocessing import Pool
from itertools import repeat as rep
import tqdm

sys.path.insert(1, os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../..')))
sys.path.insert(1, os.path.abspath(pathlib.Path(__file__).parent.resolve()))
from dutils import add_color_pixels_rand_gt, imsegkmeans, arr2tf, get_fn_wo_ext
from src.utils import files

dirs = files.config_parse(dirs=True)

ORIG_EXT = ".JPEG"

def cache_img(img, filepath, overwrite, png_compression=0):
    """
    Saves to disk, only if not already present
    """
    if not filepath.lower().endswith(('.png', '.jpg', '.jpeg')):
        filepath = filepath + ".png"
    if not os.path.isdir(os.path.dirname(filepath)):
        print("Creating dir:", os.path.dirname(filepath))
        os.makedirs(os.path.dirname(filepath))

    if not os.path.isfile(filepath) and not os.path.islink(filepath):
        cv2.imwrite(filepath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    elif overwrite:
        # TODO: load and compare to new image & save if changed
        cv2.imwrite(filepath, img, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression])
    else:
        pass


def load_img(filepath, gray=False):
    """
    :return: None if file not exists
    """
    img = None
    if (os.path.isfile(filepath) or os.path.islink(filepath) ) and filepath.lower().endswith(('.jpg', '.jpeg', ".png", ".tiff")):
        img = cv2.imread(filepath)
        rgb = False
        if filepath.lower().endswith(('.jpg', '.jpeg')):
            rgb = True
        if not gray:
            if not rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            if not rgb:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        # print("load_img; file not found:", filepath)
        pass
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

    if overwrite or theme is None or mask is None:
        gt_path = dirs[set] + dirs["ground_truth"] + filename + ".png"
        gt = load_img(gt_path)
        segmented_img, theme, mask = imsegkmeans(gt, num_points_theme)
        cache_img(theme, theme_path, overwrite)
        cache_img(mask, mask_path, overwrite)
        if save_segmented:
            cache_img(segmented_img, segmented_path, overwrite)
    return theme, mask, segmented_img
    # return arr2tf(theme), arr2tf(mask), arr2tf(segmented_img)


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

    if overwrite or points_rgb is None or points_mask is None:
        gt_path = dirs[set] + dirs["ground_truth"] + filename + ".png"
        gt = load_img(gt_path)
        points_mask, points_rgb = add_color_pixels_rand_gt(gt, num_points_pix)
        cache_img(points_mask, mask_path, overwrite)
        cache_img(points_rgb, points_path, overwrite)
    return points_rgb, points_mask
    # return arr2tf(points_rgb), arr2tf(points_mask)

def get_h_w(img):
    """
    To handle gray and color images
    """
    h = w = 0
    try:
        h, w, _ = img.shape
    except ValueError:
        h, w = img.shape
    return h, w

def crop(img, random_crop=256, fn=""):
    """
    :param fn: will be incorporated into random seed (same crop for gray & color) (w/o extension)
    :return: cropped img
    """
    if fn:
        hsh = int.from_bytes(hashlib.md5(get_fn_wo_ext(fn).encode('utf-8')).digest(), 'big') + random_crop
    else:
        hsh = np.random.randint(9999999)
    rs = np.random.RandomState(np.random.MT19937(np.random.SeedSequence(hsh)))

    h, w = get_h_w(img)
    max_sample_h = h - random_crop
    max_sample_w = w - random_crop

    sample_h = rs.randint(0, max_sample_h)
    sample_w = rs.randint(0, max_sample_w)

    return img[sample_h:sample_h + random_crop, sample_w:sample_w + random_crop]

def _gt_gen(img, filename, new_dir, random_crop=256, overwrite=False):
    h, w = get_h_w(img)
    if h <= random_crop or w <= random_crop:
        return None
    gt_path = new_dir + filename
    if overwrite or not os.path.isfile(gt_path):
        img = crop(img, random_crop, filename)
        cache_img(img, gt_path, overwrite)
        return img

def gt_gen_color(filename, set="train", random_crop=256, overwrite=False):
    """
    TODO: add align? (rotate so that larger side always width)
    :param filename: original image path (with extension)
    """
    orig_path = dirs[set] + dirs["original_img"] + filename
    img = load_img(orig_path)
    return _gt_gen(img, get_fn_wo_ext(filename), dirs[set]+dirs["ground_truth"], random_crop=random_crop, overwrite=overwrite)
    # return arr2tf(_gt_gen(img, get_fn_wo_ext(filename), set=set, random_crop=random_crop, overwrite=overwrite))

def gt_gen_gray(filename, set="train", random_crop=256, overwrite=False):
    """
    :param filename: original img path (with extension)
    """
    orig_path = dirs[set] + dirs["original_img"] + filename
    img = load_img(orig_path, gray=True)
    return _gt_gen(img, get_fn_wo_ext(filename), dirs[set]+dirs["ground_truth_gray"], random_crop=random_crop, overwrite=overwrite)
    # return arr2tf(_gt_gen(img, get_fn_wo_ext(filename), set=set, random_crop=random_crop, overwrite=overwrite))


def preprocess_grayscale(random_crop=256, set="train", overwrite=False):
    counter = 0
    cpus = int(os.cpu_count() - (os.cpu_count()/100*1))  # leave 1% of CPU for other tasks ᕙ(⇀‸↼‶)ᕗ
    os.nice(10)
    orig_img_list = []
    for dirpath, dnames, fnames in os.walk(dirs[set] + dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                orig_img_list.append(file)

    starmap_input = zip(orig_img_list, rep(set), rep(random_crop), rep(overwrite))
    with Pool(processes=cpus) as pool:
        # no wrapper needed for gray, just one call
        pool.starmap(gt_gen_gray, tqdm.tqdm(starmap_input, total=len(orig_img_list)))


def preprocess_color_once(file, num_points_pix, num_points_theme, random_crop, set,
                           locals, theme, segmented, overwrite):
    fn_wo_ext = get_fn_wo_ext(file)
    img = gt_gen_color(file, set, random_crop, overwrite)
    # if img < random crop (None), skip
    if img is None:
        return
    if locals:
        local_gen(fn_wo_ext, num_points_pix, set, overwrite)
    if theme:
        theme_gen(fn_wo_ext, set, num_points_theme, save_segmented=segmented, overwrite=overwrite)

def preprocess_color(num_points_pix, num_points_theme, random_crop=256, set="train",
                     locals=True, theme=True, segmented=True, overwrite=False):
    """
    clean python implementation of:
    https://github.com/praywj/Interactive-Deep-Colorization-and-Compression/tree/master/prepare_dataset
    iterates through "original_img" folder, to prepare color cues

    :param num_points_pix:
    :param num_points_theme:
    :param random_crop: target size of crop. If false/0, don't crop
    :param locals: gen and save locals
    :param theme: gen and save theme
    :param segmented: gen and save theme segmented (k-means)
    :param overwrite: overwrite img, if one already exists. (F.e. activate after change of point gen method)

    :return:
    """
    counter = 0
    cpus = int(os.cpu_count() - (os.cpu_count()/100*1))  # leave 1% of CPU for other tasks ᕙ(⇀‸↼‶)ᕗ
    os.nice(10)
    orig_img_list = []
    for dirpath, dnames, fnames in os.walk(dirs[set] + dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                orig_img_list.append(file)


    starmap_input = zip(orig_img_list, rep(num_points_pix), rep(num_points_theme), rep(random_crop), rep(set),
                        rep(locals), rep(theme), rep(segmented), rep(overwrite))
    with Pool(processes=cpus) as pool:
        pool.starmap(preprocess_color_once, tqdm.tqdm(starmap_input, total=len(orig_img_list)))


def dataset_prepare(set="train", max_img=None):
    """
    :param set: train, val (or test)
    :param max_img: stop after that many images. None: use all
    Links dataset into 'original_img'. Runs recursively.
    ALL images below given folder will be used.
    REMOVES all old links
    """
    # only run, if wanted number is different TODO: handle None (count recursively in dirs["dataset"]+set)
    try:
        if len(os.listdir(dirs[set] + dirs["original_img"])) == max_img:
            return
    except FileNotFoundError:
        pass

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
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                if max_img is not None and counter >= max_img:
                    break
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

def main():
    # Change here to your needs and rerun with "python src/dinterface/preprocess.py"
    # if number of max_img is reduced, they need to be deleted manually
    dataset_prepare(set="train", max_img=100_000)
    dataset_prepare(set="val", max_img=10_000)

    sets = ["train", "val"]
    for set in sets:
        preprocess_color(num_points_pix=100, num_points_theme=6, random_crop=256, set=set,
                         locals=True, theme=True, segmented=True, overwrite=False)
        preprocess_grayscale(random_crop=256, set=set, overwrite=False)


if __name__ == '__main__':
    main()
