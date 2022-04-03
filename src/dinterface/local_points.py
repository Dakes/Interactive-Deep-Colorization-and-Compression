"""
Functions to generate the local inputs. Random and targeted. + their helpers

"""

import os
import cv2
import numpy as np
import skimage
from matplotlib import pyplot as plt
from skimage.segmentation import felzenszwalb, slic
from skimage.segmentation import mark_boundaries
from scipy.stats import norm
from functools import lru_cache

from dutils import rgb_to_ab, get_error_map, plot_img

_MIN_POINTS_PIX = 10
_MAX_POINTS_PIX = 100  # h*w*0.03


def add_color_pixels_rand_gt(img, num_points):
    h, w, c = img.shape
    points_mask = np.zeros([h, w], dtype=np.uint8)
    points_rgb = np.zeros([h, w, c], dtype=np.uint8)

    # if not specified, sample uniformly distributed random number within given interval
    points = np.random.random_integers(_MIN_POINTS_PIX, _MAX_POINTS_PIX) if num_points == -1 else num_points

    # chose random points, update arrays
    for i in range(points):
        y = np.random.random_integers(0, h - 1)
        x = np.random.random_integers(0, w - 1)
        points_mask[y, x] = 255
        points_rgb[y, x] = img[y, x]

    return points_mask, points_rgb


def _segments_to_points(segmented, rgb_img, error_map, max_num_points=100):
    regions = skimage.measure.regionprops(segmented, intensity_image=error_map)

    h, w, c = rgb_img.shape
    points_mask = np.zeros([h, w], dtype=np.uint8)
    points_rgb = np.zeros([h, w, c], dtype=np.uint8)

    i = 0
    # iterate over segmented regions, sorted by mean error. Highest first.
    for props in sorted(regions, key=lambda r: r.intensity_mean[0], reverse=True):
        y, x = props.centroid  # centroid coordinates
        x, y = int(x), int(y)
        points_mask[y, x] = 255
        points_rgb[y, x] = rgb_img[y, x]
        i += 1
        if i >= max_num_points:
            # print("Skipping", len(regions) - max_num_points, "region(s)")
            break

    return points_rgb, points_mask


def get_points_slic(orig_img, theme_img, points=100, plot=False):
    error_map = get_error_map(orig_img, theme_img)

    # n_segments is the maximum number (But it almost never reaches that)
    for i in range(100):
        try:
            segments_slic = slic(error_map, compactness=0.2, sigma=1, n_segments=points, enforce_connectivity=False, convert2lab=False)
            break
        # happens if points are 0
        except ZeroDivisionError:
            points = points + 1
            print("SLIC: ZeroDivisionError. Modifying values. New Number of Points:", points)
            if i >= 100:
                exit()

    if plot:
        print(len(np.unique(segments_slic)), "segments")
        plt.imshow(mark_boundaries(error_map, segments_slic))
        plt.axis('off')
        plt.show()
    return _segments_to_points(segments_slic, orig_img, error_map, max_num_points=points)


def get_points_felzenszwalb(orig_img, theme_img, points=100, plot=False):
    error_map = get_error_map(orig_img, theme_img)

    segments_fz = felzenszwalb(error_map, scale=50, sigma=1, min_size=50)
    # slic(img, n_segments=100, compactness=5, sigma=1, start_label=1)
    if plot:
        print(len(np.unique(segments_fz)), "segments")
        plt.imshow(mark_boundaries(error_map, segments_fz))
        plt.axis('off')
        plt.show()
    return _segments_to_points(segments_fz, orig_img, error_map, max_num_points=points)


# Normal Distribution subtraction method
def _extend_dist(y):
    """
    scale normal distribution to have one as the highest value.
    """
    mul = 1 / y.max()
    for idx, val in enumerate(y):
        y[idx] = y[idx] * mul
    return y


@lru_cache()
def _get_em_subtrahend(radius=10, scale=3.5):
    x = np.arange(-radius, radius, 1)
    y = norm.pdf(x, 0, scale)
    y = _extend_dist(y)
    y2 = np.outer(y, y)
    return y2


def _subtract_from_em(em, y, x, radius=10, scale=3.5):
    """
    :param em: error map. as float 0-1.0 of shape (height, width)
    :param y, x: coordinate, where to subtract from error map
    :param radius: radius of normal distribution to subtract
    :param scale: Scale, how far distribution should spread out.
    """
    sub = _get_em_subtrahend(radius=radius, scale=scale)
    h, w = em.shape
    for yi in range(-radius, radius):
        for xi in range(-radius, radius):
            yc = y + yi
            xc = x + xi
            if yc > h - 1 or yc < 0 or xc > w - 1 or xc < 0:
                continue
            em[yc][xc] = em[yc][xc] - sub[yi + radius][xi + radius]
            # if em[yc][xc] < 0:
            #    em[yc][xc] = 0
    return em


def get_points_nodist(rgb_img, theme_img, points=100, radius=10, scale=3.5, plot=False, points_rgb=None, points_mask=None):
    """
    Choose points by subtracting normal distribution from areas with highest error and choose points there.
    :param points_rgb, points_mask: arrays to add to. If None (default), creates empty ones
    """
    em = get_error_map(rgb_img, theme_img, one_channel=True)

    h, w, c = rgb_img.shape
    if points_mask is None:
        points_mask = np.zeros([h, w], dtype=np.uint8)
    if points_rgb is None:
        points_rgb = np.zeros([h, w, c], dtype=np.uint8)

    for i in range(points):
        y, x = np.unravel_index(em.argmax(), em.shape)
        points_mask[y, x] = 255
        points_rgb[y, x] = rgb_img[y, x]
        em = _subtract_from_em(em, y, x, radius=radius, scale=scale)

    if plot:
        plot_img(em, cmap="gray")

    return points_rgb, points_mask

