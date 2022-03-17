import sys
import matplotlib.pyplot as plt
import datetime
import os
import gin
import time
import numpy as np
import tensorflow as tf

from src.compression.dinterface.dinterface import init_reading
from src.compression.compression_gan.data_utils import load_prepare_data_val, generate_images, load_norm_image, load_prepare_data_train, fit
from src.compression.compression_gan.net_architecture import make_enc, make_gen, make_gan, make_multi_scale_disc, quantizer_theis

# adjust as required
path_to_dataset = '/home/daniel/imagenet-mini'

GIN_FIN = 'extreme_compression.gin'
STRFTIME_FORMAT = "%Y%m%d-%H%M%S"

plt.rcParams['figure.figsize'] = [16, 9]


@gin.configurable('setup_optimizer')
def setup_optimizer(g_lr, d_lr, gan_loss, use_lpips, use_feature_matching):
    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=g_lr)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=d_lr)
    return generator_optimizer, discriminator_optimizer, gan_loss, use_lpips, use_feature_matching


@gin.configurable('shared_specs')
def get_shared_specs(epochs, batch_size, k_beta, k_m, k_p, k_fm, channel_bottleneck, disc_scale):
    return epochs, batch_size, k_beta, k_m, k_p, k_fm, channel_bottleneck, disc_scale


@gin.configurable('io')
def setup_io(base_path, code_directory, ckpt_dir, tb_dir, gen_imgs_dir, model_dir, log_dir, lpips_weights, input_dim_raw,
             input_dim_target, data, data_prep, buf_size):
    ckpt_path = base_path + code_directory + ckpt_dir
    tb_path = base_path + code_directory + tb_dir
    gen_path = base_path + code_directory + gen_imgs_dir
    model_path = base_path + code_directory + model_dir
    log_path = base_path + code_directory + log_dir
    lpips_path = base_path + code_directory + lpips_weights
    data = base_path + data
    data_prep = base_path + code_directory + data_prep
    return ckpt_path, tb_path, gen_path, model_path, log_path, input_dim_raw, input_dim_target, data, data_prep, lpips_path, buf_size


gin.parse_config_file(GIN_FIN)
epochs, batch_size, k_beta, k_m, k_p, k_fm, channel_bottleneck, disc_scale = get_shared_specs()
ckpt_path, tb_path, gen_path, model_path, log_path, input_dim_raw, input_dim_target, data, data_prep, lpips_path, buf_size = setup_io()

print('Experimental setting \nnumber of training epochs: {}\nbatch size: {}\nbpp: {}'.format(epochs, batch_size, 0.072 if channel_bottleneck == 8 else '?'))

# convert imagenet dataset to GAN format
if not os.path.exists(data_prep):
    print('converting to GAN format...')
    init_reading(data, data_prep, input_dim_raw)

# load and preprocess dataset
# train_ds = load_prepare_data_train(data_prep, batch_size, buf_size, input_dim_target)
#val_ds = load_prepare_data_val(data_prep, batch_size, input_dim_target)

