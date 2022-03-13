import os
import cv2
import numpy as np

from src.utils.utils import add_color_pixels_rand_gt, imsegkmeans

_BASE_PATH = '/home/nikolai10/IdeaProjects/xiao-et-al-remastered-demo/'

_IMAGE_COLOR_DIR = 'image_color_dir/'
_LOCAL_DIR = 'local_dir/'
_LOCAL_MASK_DIR = 'local_mask_dir/'
_COLOR_MAP_DIR = 'color_map_dir/'
_THEME_DIR = 'theme_dir/'
_THEME_MASK_DIR = 'theme_mask_dir/'
_IMG_SIZE = 256

_IN_DIR = _BASE_PATH + 'res/eval/image_color_dir/'
_OUT_DIR = _BASE_PATH + 'res/eval/'


def run(in_dir, out_dir, num_points_pix, num_points_theme, random_crop=True, align=False, only_locals=False):
    """
    clean python implementation of:
    https://github.com/praywj/Interactive-Deep-Colorization-and-Compression/tree/master/prepare_dataset

    :param in_dir:
    :param out_dir:
    :param num_points_pix:
    :param num_points_theme:
    :param random_crop:
    :param align:
    :return:
    """

    out_dir_rgb = out_dir + _IMAGE_COLOR_DIR
    out_dir_local = out_dir + _LOCAL_DIR
    out_dir_local_mask = out_dir + _LOCAL_MASK_DIR
    out_dir_color_map = out_dir + _COLOR_MAP_DIR
    out_dir_theme = out_dir + _THEME_DIR
    out_dir_theme_mask = out_dir + _THEME_MASK_DIR

    # create dir if not exists
    if not os.path.exists(out_dir_rgb):
        os.makedirs(out_dir_rgb)
    if not os.path.exists(out_dir_local_mask):
        os.makedirs(out_dir_local_mask)
    if not os.path.exists(out_dir_local):
        os.makedirs(out_dir_local)
    if not os.path.exists(out_dir_color_map):
        os.makedirs(out_dir_color_map)
    if not os.path.exists(out_dir_theme):
        os.makedirs(out_dir_theme)
    if not os.path.exists(out_dir_theme_mask):
        os.makedirs(out_dir_theme_mask)

    for dirpath, dnames, fnames in os.walk(in_dir):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(file)
                temp_fn = os.path.splitext(file)[0]

                item_path = os.path.join(dirpath, file)
                img = cv2.imread(item_path)
                if random_crop:
                    h, w, _ = img.shape
                    max_sample_h = h - _IMG_SIZE
                    max_sample_w = w - _IMG_SIZE
                    sample_h = np.random.random_integers(0, max_sample_h)
                    sample_w = np.random.random_integers(0, max_sample_w)
                    img = img[sample_h:sample_h + _IMG_SIZE, sample_w:sample_w + _IMG_SIZE]
                if align:
                    h, w, _ = img.shape
                    if h > w:
                        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

                if only_locals:
                    points_mask, points_rgb = add_color_pixels_rand_gt(img, num_points_pix)
                    cv2.imwrite(out_dir_local_mask + temp_fn + '.png', points_mask)
                    cv2.imwrite(out_dir_local + temp_fn + '.png', points_rgb)
                else:
                    points_mask, points_rgb = add_color_pixels_rand_gt(img, num_points_pix)
                    segmented_img, theme, mask = imsegkmeans(img, num_points_theme)
                    cv2.imwrite(out_dir_rgb + temp_fn + '.png', img)
                    cv2.imwrite(out_dir_local_mask + temp_fn + '.png', points_mask)
                    cv2.imwrite(out_dir_local + temp_fn + '.png', points_rgb)
                    cv2.imwrite(out_dir_color_map + temp_fn + '.png', segmented_img)
                    cv2.imwrite(out_dir_theme + temp_fn + '.png', theme)
                    cv2.imwrite(out_dir_theme_mask + temp_fn + '.png', mask)


def main():
    run(_IN_DIR, _OUT_DIR, num_points_pix=-1, num_points_theme=-1, random_crop=False)


if __name__ == '__main__':
    main()
