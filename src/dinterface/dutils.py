import os
import cv2
import numpy as np
import tensorflow as tf

# TODO move to config?
_MIN_POINTS_THEME = 3
_MAX_POINTS_THEME = 7
# _MIN_POINTS_PIX = 10
# _MAX_POINTS_PIX = 50
_MIN_POINTS_PIX = 10
_MAX_POINTS_PIX = 100  # h*w*0.03

# TODO: add "clever" point selection (mainly for compression)


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


def add_color_pixels_gt(img, list_regions):
    h, w, c = img.shape
    points_mask = np.zeros([h, w], dtype=np.uint8)
    points_rgb = np.zeros([h, w, c], dtype=np.uint8)

    # select centroids, update arrays
    for props in list_regions:
        y, x = props.centroid  # centroid coordinates
        x, y = int(x), int(y)
        points_mask[y, x] = 255
        points_rgb[y, x] = img[y, x]

    return points_mask, points_rgb


def imsegkmeans(rgb_img, num_points):
    """
    see https://www.thepythoncode.com/article/kmeans-for-image-segmentation-opencv-python

    :param rgb_img:
    :return: segmented image, sorted theme colors, mask
    """
    # if not specified, sample uniformly distributed random number within given interval
    points = np.random.random_integers(_MIN_POINTS_THEME, _MAX_POINTS_THEME) if num_points == -1 else num_points

    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = rgb_img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, (centers) = cv2.kmeans(pixel_values, points, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()
    # count and sort in descending order (most representative colors)
    sort_index = np.argsort(-1 * np.bincount(labels))
    sorted_centers = np.reshape(centers[sort_index], (1, points, 3))
    sorted_centers_padded = np.zeros((1, _MAX_POINTS_THEME, 3), dtype=np.uint8)
    sorted_centers_padded[:, 0:points, :] = sorted_centers

    mask = np.zeros((1, _MAX_POINTS_THEME, 1), dtype=np.uint8)
    mask[:, 0:points, :] = 255

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # reshape back to the original image dimension
    return segmented_image.reshape(rgb_img.shape), sorted_centers_padded, mask

def arr2tf(img):
    """
    assumes already loaded as RGB
    """
    return tf.keras.preprocessing.image.array_to_img(img)

def get_fn_wo_ext(fn):
    if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
        fn = os.path.splitext(fn)[0]
    return fn

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