import sys, os
import time
import pathlib
import tensorflow as tf
import numpy as np
import random
from cachier import cachier
import datetime

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
from src.colorization import model
from src.dinterface import preprocess as preprocess
from src.dinterface import dutils
from src.utils import files as files

dirs = files.config_parse(dirs=True)
pp_params = files.config_parse()["preprocess"]

'''
@cachier(stale_after=datetime.timedelta(days=1))
def get_all_paths(root_dir, ext='png', basename=False):
    """
    :return: list of paths to files in root_dir, not absolute
    """
    root_dir = pathlib.Path(root_dir)
    paths = map(str, root_dir.rglob('*.' + ext))
    if basename:
        paths = map(os.path.basename, paths)
    file_paths = list(paths)
    return file_paths

@cachier(stale_after=datetime.timedelta(days=1))
def get_new_paths(filename_list, root):
    new_paths = []
    for file in filename_list:
        new_paths.append(os.path.join(root, file).replace(".JPEG", ".png"))
    return new_paths

def _get_train_list_param_hasher(args, kwargs):
    import hashlib
    # dir_list, name_list, ext_list, shuffle=True, set="train"
    # WARNING: when shuffle=True technically it shouldn't cache it, but ¯\_(ツ)_/¯
    hashable = str(sorted(args[0])) + str(kwargs)
    return hashlib.md5(hashable.encode('utf-8')).hexdigest()

@cachier(hash_params=_get_train_list_param_hasher, stale_after=datetime.timedelta(days=1))
def get_train_list(dir_list, name_list, ext_list, shuffle=True, set="train"):
    """
    get list of /hypothetical/ paths to files, that will be generated on demand
    """
    # adjust extension to extension in dataset. Maybe add to config. Maybe change get_all_paths to match multiple.
    original_list = get_all_paths(dirs[set] + dirs["original_img"], ext="JPEG", basename=True)

    train_list = []
    for root_dir, name, ext in zip(dir_list, name_list, ext_list):
        tic = time.time()
        file_paths = sorted(get_new_paths(original_list, root_dir))
        # file_paths = sorted(get_all_paths(root_dir, ext))
        toc = time.time()
        print('[Type:%s][File nums: %d, Time_cost: %.2fs]' % (name, len(file_paths), toc - tic))
        train_list.append(np.asarray(file_paths))

    if shuffle:
        file_count = len(train_list[0])
        rnd_index = np.arange(file_count)
        np.random.shuffle(rnd_index)
        for i, item in enumerate(train_list):
            train_list[i] = item[rnd_index]

    return tuple(train_list)
'''

# @cachier(stale_after=datetime.timedelta(days=1))
def get_all_paths(root_dir, ext='png', single_img_name=None):
    root_dir = pathlib.Path(root_dir)
    glob_str = '*.' + ext
    if single_img_name:
        glob_str = dutils.get_fn_wo_ext(single_img_name) + "." + ext
    file_paths = list(map(str, root_dir.rglob(glob_str)))
    return file_paths

def get_train_list(dir_list, name_list, ext_list, shuffle=True, single_img_name=None):
    train_list = []
    for root_dir, name, ext in zip(dir_list, name_list, ext_list):
        tic = time.time()
        file_paths = sorted(get_all_paths(root_dir, ext))
        toc = time.time()
        print('[Type:%s][File nums: %d, Time_cost: %.2fs]' % (name, len(file_paths), toc - tic))
        train_list.append(np.asarray(file_paths))

    if shuffle:
        file_count = len(train_list[0])
        rnd_index = np.arange(file_count) - 2
        np.random.shuffle(rnd_index)
        for i, item in enumerate(train_list):
            train_list[i] = item[rnd_index]

    return tuple(train_list)

# both
def get_batch(train_list, image_size, batch_size, capacity, is_random=True, only_globals=False):
    tf.compat.v1.disable_eager_execution()
    print("train_list: ", train_list)

    filepath_queue = tf.compat.v1.train.slice_input_producer(train_list, shuffle=False)
    # filepath_queue = tf.data.Dataset.from_tensor_slices(tuple(train_list))
    # print("\nTrain_list:", train_list)
    # print("\nfilepath_queue:", filepath_queue.numpy())
    # print("\nfilepath_queue STR ===:", filepath_queue[0].numpy() )
    # fn_wo_ext = dutils.get_fn_wo_ext(filepath_queue[0])
    img_size_h, img_size_w = image_size

    # color
    image_rgb = tf.io.read_file(filepath_queue[0])
    image_rgb = tf.image.decode_png(image_rgb, channels=3)
    # image_rgb = preprocess.gt_gen_color(filepath_queue[0], set="train", random_crop=image_size)
    image_rgb = tf.image.resize(image_rgb, [img_size_h, img_size_w])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.  # 归一化到[0, 1]

    # Theme
    # theme_rgb, theme_mask, index_rgb = preprocess.theme_gen(fn_wo_ext, set="train", num_points_theme=6)
    # color theme
    theme_rgb = tf.io.read_file(filepath_queue[1])
    theme_rgb = tf.image.decode_png(theme_rgb, channels=3)
    theme_rgb = tf.image.resize(theme_rgb, [1, 7])
    theme_rgb = tf.cast(theme_rgb, tf.float32) / 255.

    # color theme mask
    theme_mask = tf.io.read_file(filepath_queue[2])
    theme_mask = tf.image.decode_png(theme_mask, channels=1)
    theme_mask = tf.image.resize(theme_mask, [1, 7])
    theme_mask = tf.cast(theme_mask, tf.float32) / 255.
    theme_mask = tf.reshape(theme_mask[:, :, 0], [1, 7, 1])

    # a K-color map by decoding the color image with its representative colors
    index_rgb = tf.io.read_file(filepath_queue[3])
    index_rgb = tf.image.decode_png(index_rgb, channels=3)
    index_rgb = tf.image.resize(index_rgb, [img_size_h, img_size_w])
    index_rgb = tf.cast(index_rgb, tf.float32) / 255.

    # Local Points
    # point_rgb, point_mask = preprocess.local_gen(fn_wo_ext, num_points_pix=-1, set="train")  # -1: random number
    # local rgb
    point_rgb = tf.io.read_file(filepath_queue[4])
    point_rgb = tf.image.decode_png(point_rgb, channels=3)
    point_rgb = tf.image.resize(point_rgb, [img_size_h, img_size_w])
    point_rgb = tf.cast(point_rgb, tf.float32) / 255.

    # local mask
    point_mask = tf.io.read_file(filepath_queue[5])
    point_mask = tf.image.decode_png(point_mask, channels=1)
    point_mask = tf.image.resize(point_mask, [img_size_h, img_size_w])
    point_mask = tf.cast(point_mask, tf.float32) / 255.
    point_mask = tf.reshape(point_mask[:, :, 0], [img_size_h, img_size_w, 1])

    # set to zero
    point_rgb_blank = tf.zeros([img_size_h, img_size_w, 3], dtype=tf.float32)
    point_mask_blank = tf.zeros([img_size_h, img_size_w, 1], dtype=tf.float32)

    # set to zero by random
    rnd = tf.random.uniform(shape=[1], minval=0, maxval=1, dtype=tf.float32)
    rnd = rnd[0]

    def f1():  # only global
        return theme_rgb, theme_mask, index_rgb, point_rgb_blank, point_mask_blank

    def f3():  # both
        return theme_rgb, theme_mask, index_rgb, point_rgb, point_mask

    # 5% use only global, else both global and local inputs
    if is_random is True:
        rate1 = 0.05
        flag1 = tf.less(rnd, rate1)
        flag3 = tf.greater_equal(rnd, rate1)
        theme_rgb, theme_mask, index_rgb, point_rgb, point_mask = \
            tf.case({flag1: f1, flag3: f3}, exclusive=True)

    if is_random is True:
        image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
            tf.compat.v1.train.shuffle_batch(
                [image_rgb, theme_rgb, theme_mask, index_rgb, point_rgb, point_mask],
                batch_size=batch_size,
                capacity=capacity,
                min_after_dequeue=500,
                num_threads=16)
    else:
        if only_globals:
            point_rgb = point_rgb_blank
            point_mask = point_mask_blank

        image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
            tf.compat.v1.train.batch(
                [image_rgb, theme_rgb, theme_mask, index_rgb, point_rgb, point_mask],
                 batch_size=1,
                 capacity=capacity,
                 num_threads=1)

    return image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch


# Convert from RGB space to LAB space, require normalized RGB input, and unnormalized output LAB, require float32
def rgb_to_lab(image_rgb):
    assert image_rgb.get_shape()[-1] == 3

    rgb_pixels = tf.reshape(image_rgb, [-1, 3])
    # RGB to XYZ
    with tf.compat.v1.name_scope("rgb_to_xyz"):
        linear_mask = tf.cast(rgb_pixels <= 0.04045, dtype=tf.float32)
        expoential_mask = tf.cast(rgb_pixels > 0.04045, dtype=tf.float32)
        rgb_pixels = (rgb_pixels / 12.92) * linear_mask + \
                     (((rgb_pixels + 0.055) / 1.055) ** 2.4) * expoential_mask
        transfer_mat = tf.constant([
            [0.412453, 0.212671, 0.019334],
            [0.357580, 0.715160, 0.119193],
            [0.180423, 0.072169, 0.950227]
        ], dtype=tf.float32)
        xyz_pixels = tf.matmul(rgb_pixels, transfer_mat)

    # XYZ to LAB
    with tf.compat.v1.name_scope("xyz_to_lab"):
        # Standardized D65 white point
        xyz_norm_pixels = tf.multiply(xyz_pixels, [1 / 0.950456, 1.0, 1 / 1.088754])
        epsilon = 6 / 29
        linear_mask = tf.cast(xyz_norm_pixels <= epsilon ** 3, dtype=tf.float32)
        expoential_mask = tf.cast(xyz_norm_pixels > epsilon ** 3, dtype=tf.float32)
        f_xyf_pixels = (xyz_norm_pixels / (3 * epsilon ** 2) + 4 / 29) * linear_mask + \
                       (xyz_norm_pixels ** (1 / 3)) * expoential_mask
        transfer_mat2 = tf.constant([
            [0.0, 500.0, 0.0],
            [116.0, -500.0, 200.0],
            [0.0, 0.0, -200.0]
        ], dtype=tf.float32)
        lab_pixels = tf.matmul(f_xyf_pixels, transfer_mat2) + tf.constant([-16.0, 0.0, 0.0], dtype=tf.float32)

        image_lab = tf.reshape(lab_pixels, tf.shape(input=image_rgb))

    return image_lab


# LAB space to RGB space
def lab_to_rgb(image_lab):
    assert image_lab.shape[-1] == 3

    lab_pixels = tf.reshape(image_lab, [-1, 3])
    with tf.compat.v1.name_scope('lab_to_xyz'):
        transfer_mat1 = tf.constant([
            [1 / 116.0, 1 / 116.0, 1 / 116.0],
            [1 / 500.0, 0.0, 0.0],
            [0.0, 0.0, -1 / 200.0]
        ], dtype=tf.float32)
        fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), transfer_mat1)
        epsilon = 6 / 29
        linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
        expoential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
        xyz_pixels = (3 * epsilon ** 2 * (fxfyfz_pixels - 4 / 29)) * linear_mask + \
                     (fxfyfz_pixels ** 3) * expoential_mask
        xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

    with tf.compat.v1.name_scope('xyz_to_rgb'):
        transfer_mat2 = tf.constant([
            [3.2404542, -0.9692660, 0.0556434],
            [-1.5371385, 1.8760108, -0.2040259],
            [-0.4985314, 0.0415560, 1.0572252]
        ])
        rgb_pixels = tf.matmul(xyz_pixels, transfer_mat2)
        rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
        linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
        expoential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
        rgb_pixels = rgb_pixels * 12.92 * linear_mask + \
                     ((rgb_pixels ** (1 / 2.4) * 1.055) - 0.055) * expoential_mask

        image_rgb = tf.reshape(rgb_pixels, tf.shape(input=image_lab))

    return image_rgb


def get_eval_img(img_path, theme_path, theme_mask_path, point_path, point_mask_path, img_size=(256, 256),
                 only_globals=False):
    max_iter = 1_000
    fn = None
    i = 0
    while True:
        fn = random.choice(os.listdir(img_path))
        i += 1
        if i > max_iter:
            print("(Probably) no images in val dir:", img_path)
            exit()
        if fn.lower().endswith((".png", ".jpg", ".jpeg", ".tiff")):
            break

    img_path = img_path + fn
    theme_path = theme_path + fn
    theme_mask_path = theme_mask_path + fn
    point_path = point_path + fn
    point_mask_path = point_mask_path + fn

    h, w = img_size
    image_rgb = tf.io.read_file(img_path)
    image_rgb = tf.image.decode_png(image_rgb, channels=3)
    image_rgb = tf.image.resize(image_rgb, [h, w])
    image_rgb = tf.cast(image_rgb, tf.float32) / 255.  # 归一化到[0, 1]
    image_rgb = tf.reshape(image_rgb, [1, h, w, 3])

    theme_rgb = tf.io.read_file(theme_path)
    theme_rgb = tf.image.decode_png(theme_rgb, channels=3)
    theme_rgb = tf.image.resize(theme_rgb, [1, 7])
    theme_rgb = tf.cast(theme_rgb, tf.float32) / 255.
    theme_rgb = tf.reshape(theme_rgb, [1, 1, 7, 3])

    theme_mask = tf.io.read_file(theme_mask_path)
    theme_mask = tf.image.decode_png(theme_mask, channels=1)
    theme_mask = tf.image.resize(theme_mask, [1, 7])
    theme_mask = tf.cast(theme_mask, tf.float32) / 255.
    theme_mask = tf.reshape(theme_mask[:, :, 0], [1, 1, 7, 1])

    if only_globals:
        point_rgb = tf.zeros([1, h, w, 3], dtype=tf.float32)
        point_mask = tf.zeros([1, h, w, 1], dtype=tf.float32)
    else:
        point_rgb = tf.io.read_file(point_path)
        point_rgb = tf.image.decode_png(point_rgb, channels=3)
        point_rgb = tf.image.resize(point_rgb, [h, w])
        point_rgb = tf.cast(point_rgb, tf.float32) / 255.
        point_rgb = tf.reshape(point_rgb, [1, h, w, 3])

        point_mask = tf.io.read_file(point_mask_path)
        point_mask = tf.image.decode_png(point_mask, channels=1)
        point_mask = tf.image.resize(point_mask, [h, w])
        point_mask = tf.cast(point_mask, tf.float32) / 255.
        point_mask = tf.reshape(point_mask[:, :, 0], [1, h, w, 1])

    # TODO: 颜色空间转换
    image_lab = rgb_to_lab(image_rgb)
    image_l = image_lab[:, :, :, 0] / 100. * 2 - 1  # 归一化到[-1, 1]之间
    image_l = tf.reshape(image_l, [1, h, w, 1])
    image_l_gra = model.sobel(image_l)

    theme_lab = rgb_to_lab(theme_rgb)
    theme_ab = (theme_lab[:, :, :, 1:] + 128) / 255. * 2 - 1  # 归一化到[-1, 1]之间

    point_lab = rgb_to_lab(point_rgb)
    point_ab = (point_lab[:, :, :, 1:] + 128) / 255. * 2 - 1  # 归一化到[-1, 1]之间

    return image_l, theme_ab, theme_mask, point_ab, point_mask, image_rgb, image_l_gra
