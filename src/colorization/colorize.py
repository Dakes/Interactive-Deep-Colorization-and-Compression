#!/usr/bin/env python3

"""
Extracts color theme and individual pixels, for best recolorization.
Saves them and can recolor grayscale images using these color hints. (Also grayscale compressed ones)

"""

import os, sys
import argparse
import pathlib
import cv2
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio

sys.path.insert(1, os.path.join(sys.path[0], '../..'))
import src.dinterface.dutils as dutils
import src.dinterface.preprocess as preprocess
import src.utils.files as files
from src.colorization import input_data, model
from src.dinterface import local_points

dirs = files.config_parse(dirs=True)


class Colorizer(object):
    def __init__(self, num_points=6, num_points_pix=-1, random_crop=256) -> None:
        self.num_points_theme = num_points
        self.num_points_pix = num_points_pix
        self.rancom_crop = random_crop
        self.dirs = files.config_parse(dirs=True)
        self.ext = ".png"
        self.shape = (256, 256)
        # lower CPU priority (to not freeze PC), unix only
        os.nice(10)

        self.save_segmented = True

    def main(self):
        parser = argparse.ArgumentParser(
            prog="Colorizer",
            description="Recolorizes images using Xiao et al's  colorization system")
        parser.add_argument(
            "-r", "--recolorize", action="store_true", dest="decompress", type=str,
            help="Recolorize only. ")
        parser.add_argument(
            "-c", "--compress", action="store_true", dest="compress", type=str,
            help="Compress (save color cues) only. ")

        self.args = parser.parse_args()
        self.recolorize = self.args.recolorize
        self.compress = self.args.compress


    def color_cue_gen(self, rgb_fp=False, gt_gen=True, overwrite=False):
        """
        Generate color cues.
        :param rgb_fp: Path to one image file, or false, to process whole folder (res/img/test/original_img/)
        :param gt_gen: False: gt given in right folder, instead of original. skip gt_gen.
        """
        if rgb_fp:
            rgb_fp = os.path.basename(rgb_fp)
            if gt_gen:
                preprocess.gt_gen_color(rgb_fp, set="test_img", random_crop=False, overwrite=overwrite)
            preprocess.local_gen(rgb_fp, num_points_pix=self.num_points_pix, set="test_img", overwrite=overwrite)
            preprocess.theme_gen(rgb_fp, set="test_img", num_points_theme=self.num_points_theme,
                                 save_segmented=self.save_segmented, overwrite=overwrite)
        else:
            preprocess.preprocess_color(num_points_pix=self.num_points_pix, num_points_theme=self.num_points_theme,
                                        random_crop=self.rancom_crop, set="test_img",
                                        ground_truth=gt_gen, locals=True, theme=True, segmented=True, overwrite=overwrite)


    def smart_point_choice(self, method="nodist"):
        # reset local cues, to make a global only colorization, to create error map
        preprocess.preprocess_color(num_points_pix=0, num_points_theme=self.num_points_theme,
                                    random_crop=self.rancom_crop, set="test_img",
                                    ground_truth=False, locals=True, theme=False, segmented=False,
                                    overwrite=True)
        # self.recolorize_all(shape=None)
        self.recolorize()

        rec_path = self.dirs["final"] + self.dirs["recolorized"]
        directory = os.fsencode(rec_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(self.ext):
                gt = dutils.load_img(self.dirs["test_img"]+self.dirs["ground_truth"] + filename)
                theme_rec = dutils.load_img(self.dirs["final"] + self.dirs["recolorized"] + filename)
                # generate & save new points
                points_rgb, points_mask = None, None
                if method == "nodist":
                    points_rgb, points_mask = local_points.get_points_nodist(gt, theme_rec)
                elif method == "slic":
                    points_rgb, points_mask = local_points.get_points_slic(gt, theme_rec, points=100, plot=False)
                elif method == "felzenszwalb":
                    points_rgb, points_mask = local_points.get_points_felzenszwalb(gt, theme_rec, points=100, plot=False)
                else:
                    print("Error method", method, "not valid. Exiting. ")
                    exit()
                preprocess.cache_img(points_rgb, dirs["test_img"] + dirs["local_hints"] + filename, overwrite=True)
                preprocess.cache_img(points_mask, dirs["test_img"] + dirs["local_mask"] + filename, overwrite=True)


    def recolorize(self, fp="", shape=(256, 256)):
        """
        :param fp: path/name of image to recolorize. If None processes whole folder. (Requires shape then)
        :param shape: shape to resize to. None: use image size
        """
        save_path = self.dirs["final"] + self.dirs["recolorized"]
        ckpt_dir = self.dirs["colorization_model"]
        fp = self.dirs["test_img"] + self.dirs["ground_truth"] + dutils.get_fn_wo_ext(os.path.basename(fp)) + self.ext
        h, w = (256, 256)
        if shape is not None:
            h, w = shape
        if fp and shape is None:
            img = cv2.imread(fp)
            if img is None:
                print("Image not found:", fp)
                return
            h, w = dutils.get_h_w(img)
        capacity = 1000
        batch_size = 1

        # because of migration from tf 1.x -> 2.x
        tf.compat.v1.disable_eager_execution()
        sess = tf.compat.v1.Session()
        train_list = input_data.get_train_list(
            [
                self.dirs["test_img"]+self.dirs["ground_truth"],
                self.dirs["test_img"]+self.dirs["theme_rgb"], self.dirs["test_img"]+self.dirs["theme_mask"], self.dirs["test_img"]+self.dirs["color_map"],
                self.dirs["test_img"]+self.dirs["local_hints"], self.dirs["test_img"]+self.dirs["local_mask"]
            ],
            ['gt img', 'theme img', 'theme mask', 'color_map img', 'local img', 'local mask'],
            ['png', 'png', 'png', 'png', 'png', 'png'], shuffle=False, single_img_name=os.path.basename(fp))

        image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
            input_data.get_batch(train_list, (h, w), batch_size, capacity, False)

        image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
        image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [batch_size, h, w, 1])
        image_l_gra_batch = model.sobel(image_l_batch)

        theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
        theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

        point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
        point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

        # TODO: colorization
        out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch,
                                          point_ab_batch, point_mask_batch,
                                          is_training=False, scope_name='UserGuide')
        # TODO: residual network
        _, out_ab_batch2 = model.gen_PRLNet(out_ab_batch, image_l_batch, 2, scope_name='PRLNet')
        test_rgb_out2 = \
            input_data.lab_to_rgb(
                tf.concat([(image_l_batch + 1.) / 2 * 100., (out_ab_batch2 + 1.) / 2 * 255. - 128], axis=3))

        var_list = tf.compat.v1.global_variables()
        var_model1 = [var for var in var_list if var.name.startswith('UserGuide')]
        var_model2 = [var for var in var_list if var.name.startswith('PRLNet')]
        var_total = var_model1 + var_model2
        paras_count1 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model1])
        paras_count2 = tf.reduce_sum([tf.reduce_prod(v.shape) for v in var_model2])
        print('UserGuide Number of parameters 参数数目:%d' % sess.run(paras_count1))
        print('Detailed Number of parameters 参数数目:%d' % sess.run(paras_count2))

        saver1 = tf.compat.v1.train.Saver(var_list=var_total)
        print('Load checkpoint 载入检查点...')
        ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        if ckpt and ckpt.model_checkpoint_path:
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            saver1.restore(sess, ckpt.model_checkpoint_path)
            print('Loaded successfully 载入成功, global_step = %s' % global_step)
        else:
            print('Loaded successfully 载入失败')

        coord = tf.train.Coordinator()
        threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)
        # compute the average psnr
        avg_psnr = 0.
        avg_ms_ssim = 0.
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        try:
            for t in range(len(train_list[0])):
                in_rgb, out_rgb2 = sess.run([image_rgb_batch, test_rgb_out2])
                in_rgb = in_rgb[0]
                out_rgb2 = out_rgb2[0]
                psnr = peak_signal_noise_ratio(out_rgb2, in_rgb)
                avg_psnr += psnr
                plt.imsave(save_path + '/' + train_list[0][t].split('/')[-1], out_rgb2)
                # print('%s' % str(psnr))

            print('avg_psnr = %s' % str(avg_psnr / len(train_list[0])))

        except tf.errors.OutOfRangeError:
            print('Done.')
        finally:
            coord.request_stop()

        # Wait for the thread to end
        coord.join(threads=threads)
        sess.close()

    # LEGACY garbage, recolorize already does that, ups
    def recolorize_all(self, shape=(256, 256), method="nodist"):
        """
        :param shape: if None, /try/ to use original shape. Else reshape to this
        :param method: method for smart point choice. Will do (at least) one recolorization, to choose better points.
        "nodist", "slic" or "felzenszwalb". If None uses random points.
        """
        gt_path = self.dirs["test_img"] + self.dirs["ground_truth"]
        directory = os.fsencode(gt_path)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(self.ext):
                try:
                    self.recolorize(fp=filename, shape=shape)
                except ValueError as err:
                    print(err)


if __name__ == "__main__":
    # os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    c = Colorizer(num_points=6, num_points_pix=100, random_crop=256)
    # c.main()
    c.color_cue_gen(overwrite=False, gt_gen=False)
    c.smart_point_choice(method="nodist")
    c.recolorize()
    # c.recolorize_all(shape=None)
    # c.recolorize("res/img/test/original_img/20191207_115205.jpg")

