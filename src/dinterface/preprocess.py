import os
import cv2
import numpy as np

from src.dinterface.utils import add_color_pixels_rand_gt, imsegkmeans
import src.utils.files as files


_IMG_SIZE = 256


def preprocess(dirs, num_points_pix, num_points_theme, random_crop=True, align=False, only_locals=False):
    """
    clean python implementation of:
    https://github.com/praywj/Interactive-Deep-Colorization-and-Compression/tree/master/prepare_dataset

    :param dirs: config parser (dict-like) with directories
    :param num_points_pix:
    :param num_points_theme:
    :param random_crop:
    :param align:
    :return:
    """
    if not dirs:
        dirs = files.config_parse(dirs=True)

    out_dir_rgb = dirs["original_img"]
    out_dir_local = dirs["local_hints"]
    out_dir_local_mask = dirs["local_mask"]
    out_dir_color_map = dirs["color_map"]
    out_dir_theme = dirs["theme_rgb"]
    out_dir_theme_mask = dirs["theme_mask"]

    for dirpath, dnames, fnames in os.walk(dirs["original_img"]):
        for file in fnames:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # print(file)
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
    run(dirs=None, num_points_pix=-1, num_points_theme=-1, random_crop=False)


if __name__ == '__main__':
    main()
