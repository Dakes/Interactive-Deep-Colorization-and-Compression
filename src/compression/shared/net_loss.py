import tensorflow as tf
import os
import urllib.request
import numpy as np

_LPIPS_URL = "http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb"
_GAN_LOSSES = ['not_saturating', 'least_squares']


def gen_loss(d_fake_logits, d_x_acts, d_gz_acts, gen_output, target, k_beta, k_m, k_p, k_fm, lpips_path, use_lpips,
             use_feature_matching, warm_up, gan_loss='not_saturating'):
    """
    Returns the non-saturating generator loss based on
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py

    Loss := k_m * MSE + [k_p * LPIPS] + [k_fm * FM_Loss] + k_beta * GAN_Loss

    :param d_fake_logits:
    :param d_x_acts:
    :param d_gz_acts:
    :param gen_output:
    :param target:
    :param k_beta:
    :param k_m:
    :param k_p:
    :param k_fm:
    :param lpips_path:
    :param use_lpips:
    :param use_feature_matching:
    :param warm_up:
    :param gan_loss:
    :return:
    """
    # compute multi scale G loss
    multi_g_loss = []

    # minimize Jensen-Shannon divergence
    if gan_loss == _GAN_LOSSES[0]:
        for d_fake_logits_i in d_fake_logits:
            g_loss_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits_i,
                                                                              labels=tf.ones_like(d_fake_logits_i),
                                                                              name="cross_entropy_g"))
            multi_g_loss.append(g_loss_i)

    # minimize Pearson \chi2 divergence
    elif gan_loss == _GAN_LOSSES[1]:
        for d_fake_logits_i in d_fake_logits:
            g_loss_i = tf.reduce_mean(tf.square(d_fake_logits_i - 1))
            multi_g_loss.append(g_loss_i)

    # implement more GAN losses if required
    else:
        raise ValueError('GAN loss not supported, please have a look at src.generative.net_loss for more information')

    g_loss = tf.math.reduce_sum(multi_g_loss)

    # compute individual loss terms
    l2_loss = tf.reduce_mean(tf.square(target - gen_output))
    lpips_loss = compute_perceptual_loss(target, gen_output, lpips_path)
    fm_loss = 0 # compute_feature_matching_loss(d_x_acts, d_gz_acts)

    # warm-up phase
    total_g_loss = k_m * l2_loss
    if use_lpips:
        total_g_loss += k_p * lpips_loss
    if use_feature_matching:
        total_g_loss += k_fm * fm_loss

    # full learning objective
    if not warm_up:
        total_g_loss += k_beta * g_loss

    '''
    # warm-up phase (only L2)
    total_g_loss = k_m * l2_loss

    # full learning objective
    if not warm_up:
        total_g_loss += k_beta * g_loss
        if use_feature_matching:
            total_g_loss += k_fm * fm_loss
        if use_lpips:
            total_g_loss += k_p * lpips_loss
    '''
    return g_loss, l2_loss, lpips_loss, fm_loss, total_g_loss


def disc_loss(d_real_logits, d_fake_logits, gan_loss='not_saturating'):
    """
    Returns the non-saturating (multi scale) discriminator loss based on
    https://github.com/google/compare_gan/blob/master/compare_gan/gans/loss_lib.py

    # TODO add other gan losses

    :param disc_real_output:
    :param disc_generated_output:
    :param gan_loss:
    :return:
    """
    multi_scale_real = []
    multi_scale_fake = []

    # minimize Jensen-Shannon divergence
    if gan_loss == _GAN_LOSSES[0]:
        for d_real_logits_i, d_fake_logits_i in zip(d_real_logits, d_fake_logits):
            d_loss_real_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real_logits_i,
                                                                                   labels=tf.ones_like(d_real_logits_i),
                                                                                   name="cross_entropy_d_real"))
            d_loss_fake_i = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake_logits_i,
                                                                                   labels=tf.zeros_like(
                                                                                       d_fake_logits_i),
                                                                                   name="cross_entropy_d_fake"))
            multi_scale_real.append(d_loss_real_i)
            multi_scale_fake.append(d_loss_fake_i)

    # minimize Pearson \chi2 divergence
    elif gan_loss == _GAN_LOSSES[1]:
        for d_real_logits_i, d_fake_logits_i in zip(d_real_logits, d_fake_logits):
            d_loss_real_i = tf.reduce_mean(tf.square(d_real_logits_i - 1))
            d_loss_fake_i = tf.reduce_mean(tf.square(d_fake_logits_i))

            multi_scale_real.append(d_loss_real_i)
            multi_scale_fake.append(d_loss_fake_i)

    # implement more GAN losses if required
    else:
        raise ValueError('GAN loss not supported, please have a look at src.generative.net_loss for more information')

    d_loss_real = tf.math.reduce_sum(multi_scale_real)
    d_loss_fake = tf.math.reduce_sum(multi_scale_fake)
    d_loss = d_loss_real + d_loss_fake
    return d_loss, d_loss_real, d_loss_fake


def ensure_lpips_weights_exist(weight_path_out):
    if os.path.isfile(weight_path_out):
        return
    print("Downloading LPIPS weights:", _LPIPS_URL, "->", weight_path_out)
    urllib.request.urlretrieve(_LPIPS_URL, weight_path_out)
    if not os.path.isfile(weight_path_out):
        raise ValueError(f"Failed to download LPIPS weights from {_LPIPS_URL} "
                         f"to {weight_path_out}. Please manually download!")


def compute_feature_matching_loss(d_x_acts, d_gz_acts):
    """based on https://arxiv.org/abs/1711.11585"""
    return tf.reduce_sum([tf.reduce_mean(tf.abs(di_act - di_G_act)) for di_act, di_G_act in
                          zip(np.array(d_x_acts).flatten(), np.array(d_gz_acts).flatten())])


def compute_perceptual_loss(x, x_hat, lpips_path):
    """based on https://github.com/tensorflow/compression/blob/master/models/hific/model.py"""

    # First the fake images, then the real! Otherwise no gradients.
    return LPIPSLoss(lpips_path)(x_hat, x)


class LPIPSLoss(object):
    """
    Calculate LPIPS loss based on:
    https://github.com/tensorflow/compression/blob/master/models/hific/model.py

    call: lpips_loss = LPIPSLoss(_lpips_weight_path)
    """

    def __init__(self, weight_path):
        ensure_lpips_weights_exist(weight_path)

        def wrap_frozen_graph(graph_def, inputs, outputs):
            def _imports_graph_def():
                tf.graph_util.import_graph_def(graph_def, name="")

            wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
            import_graph = wrapped_import.graph
            return wrapped_import.prune(
                tf.nest.map_structure(import_graph.as_graph_element, inputs),
                tf.nest.map_structure(import_graph.as_graph_element, outputs))

        # Pack LPIPS network into a tf function
        graph_def = tf.compat.v1.GraphDef()
        with open(weight_path, "rb") as f:
            graph_def.ParseFromString(f.read())
        self._lpips_func = tf.function(
            wrap_frozen_graph(
                graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

    def __call__(self, fake_image, real_image):
        """
        Assuming inputs are in [-1, 1].

        :param fake_image:
        :param real_image:
        :return:
        """

        # Move inputs to NCHW format.
        def _transpose_to_nchw(x):
            return tf.transpose(x, (0, 3, 1, 2))

        fake_image = _transpose_to_nchw(fake_image)
        real_image = _transpose_to_nchw(real_image)
        if fake_image.shape[1] == 1:
            # broadcasting the grayscale image of dim (1, 1, 256, 256) to (1, 3, 256, 256) to use lpips on 3 identical
            # "color" channels
            fake_image = tf.broadcast_to(fake_image,[1, 3, 256, 256])
            real_image = tf.broadcast_to(real_image, [1, 3, 256, 256])
        loss = self._lpips_func(fake_image, real_image)
        return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.