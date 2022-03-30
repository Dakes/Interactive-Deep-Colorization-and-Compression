import time

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.misc
import numpy as np
from src.compression.shared.net_architecture import quantizer_theis
from src.compression.shared.net_loss import gen_loss, disc_loss


def load_prepare_data_val(input_dir, gray=False):
    """
    based on https://www.tensorflow.org/tutorials/generative/pix2pix

    :param input_dir:
    :param gray:
    :return:
    """
    # mode = 'train' if full_res else 'valid'

    if gray:
        val_dataset = tf.data.Dataset.list_files(input_dir + 'val/ground_truth_gray/*.png')
    else:
        val_dataset = tf.data.Dataset.list_files(input_dir + 'val/ground_truth/*.png')
    val_dataset = val_dataset.map(lambda x: load_norm_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(1)

    return val_dataset


def load_norm_image(image_file):
    input_image_fn = tf.io.read_file(image_file)
    input_image = tf.image.decode_png(input_image_fn)
    input_image = tf.cast(input_image, tf.float32)

    # randomly cropping to input_dim_target
    #if not mode == 'train':
         #input_image = tf.image.random_crop(input_image, size=input_dim_target)

    # normalize images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    return input_image


def generate_images(encoder, decoder, example_input, gray=False):
    w = encoder(example_input, training=False)
    z = quantizer_theis(w)
    x_hat = decoder(z, training=False)

    plt.figure(figsize=(15, 15))

    display_list = [example_input[0], x_hat[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        if gray:
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def load_prepare_data_train(input_dir, batch_size, buf_size, gray=False):
    """
    based on https://www.tensorflow.org/tutorials/generative/pix2pix
    :param input_dir:
    :param batch_size:
    :param buf_size:
    :param input_dim_target:
    :return:
    """

    if gray:
        train_dataset = tf.data.Dataset.list_files(input_dir + 'train/ground_truth_gray/*.png')
    else:
        train_dataset = tf.data.Dataset.list_files(input_dir + 'train/ground_truth/*.png')
    train_dataset = train_dataset.map(lambda x: load_norm_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(buf_size)
    train_dataset = train_dataset.batch(batch_size)
    # allow later elements to be prepared while the current element is being processed (improve latency + throughput)
    train_dataset = train_dataset.prefetch(batch_size * 2)

    return train_dataset


def generate_test_image(encoder, decoder, example_input, out_fn, gray=False):
    input = load_norm_image(example_input)
    shapes = input.shape

    input_4d = np.asarray(input).reshape((1, shapes[0], shapes[1], shapes[2]))

    w = encoder(input_4d, training=False)
    z = quantizer_theis(w)
    x_hat = decoder(z, training=False)

    out = x_hat[0]


    if gray:
        plt.imsave(out_fn, out[:,:,0] * 0.5 + 0.5, cmap='gray')
    else:
        plt.figure(figsize=(15, 15))

        display_list = [input, x_hat[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i + 1)
            plt.title(title[i])
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.savefig(out_fn)

"""
    plt.figure(figsize=(15, 15))
    plt.imshow(x_hat[0] * 0.5 + 0.5)
    plt.savefig(out_fn)
    """

"""
def generate_test_image(encoder, decoder, example_input, out_fn, gray=False):
    w = encoder(example_input, training=False)
    z = quantizer_theis(w)
    x_hat = decoder(z, training=False)

    plt.figure(figsize=(15, 15))

    display_list = [example_input[0], x_hat[0]]
    title = ['Input Image', 'Predicted Image']

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.title(title[i])
        if gray:
            plt.imshow(display_list[i] * 0.5 + 0.5, cmap='gray')
        else:
            plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.savefig(out_fn)
"""
def load_prepare_data_test(input_dir):

    val_dataset = tf.data.Dataset.list_files(input_dir + '/*.png')
    val_dataset = val_dataset.map(lambda x: load_norm_image(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.batch(1)

    return val_dataset

"""
def load_norm_image(image_file, input_dim_target, mode='train'):
    input_image_fn = tf.io.read_file(image_file)
    input_image = tf.image.decode_png(input_image_fn)
    input_image = tf.cast(input_image, tf.float32)

    # randomly cropping to input_dim_target
    if not mode == 'train':
        input_image = tf.image.random_crop(input_image, size=input_dim_target)

    # normalize images to [-1, 1]
    input_image = (input_image / 127.5) - 1
    return input_image
"""

def fit(train_ds, val_ds, epochs, ckpt, ckpt_prefix, gan, encoder, decoder, disc, sum_writer, enc_dec_opt, disc_opt,
        gan_loss, k_beta, k_m, k_p, k_fm, gen_path, lpips_path, model_path, use_lpips, use_feature_matching, gray=False,
        warm_up=False, load_ckpts=True):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(gen_path):
        os.makedirs(gen_path)

    if load_ckpts:
        # restore and continue training; else start from scratch
        ckpt_path = os.path.dirname(ckpt_prefix)
        ckpt.restore(tf.train.latest_checkpoint(ckpt_path))
        if tf.train.latest_checkpoint(ckpt_path):
            print("Restored from {}".format(tf.train.latest_checkpoint(ckpt_path)))
        else:
            print("Initializing from scratch.")

    for epoch in range(epochs):
        start = time.time()

        for example_input in val_ds.take(1):
            generate_images(encoder, decoder, example_input, gray)

        print("Epoch: ", epoch)

        # Train
        for n, (input_image) in train_ds.enumerate():
            print('.', end='')
            if (n + 1) % 100 == 0:
                print()
            train_step(input_image, gan, disc, sum_writer, enc_dec_opt, disc_opt, gan_loss, k_beta, k_m, k_p, k_fm,
                       lpips_path, use_lpips, use_feature_matching, warm_up)
        print()

        # saving (checkpoint) the model every 20 epochs
        if (epoch + 1) % 20 == 0:
            ckpt.save(file_prefix=ckpt_prefix)

        print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time() - start))
    ckpt.save(file_prefix=ckpt_prefix)


@tf.function
def train_step(img, gan, disc, sm_writer, enc_dec_opt, disc_opt, gan_loss, k_beta, k_m, k_p, k_fm, lpips_path,
               use_lpips, use_feature_matching, warm_up):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        x_hat, z, d_gz_outs, d_gz_acts = gan(img, training=True)
        d_x_outs, d_x_acts = disc([z, img], training=True)

        # compute losses
        g_loss, l2_loss, lpips_loss, lfm_loss, g_loss_total = gen_loss(d_gz_outs, d_x_acts, d_gz_acts, x_hat, img,
                                                                       k_beta, k_m, k_p, k_fm, lpips_path, use_lpips,
                                                                       use_feature_matching, warm_up, gan_loss)
        d_loss, d_loss_real, d_loss_fake = disc_loss(d_x_outs, d_gz_outs, gan_loss)

    # skip Discriminator update in warm_up mode
    if not warm_up:
        disc.trainable = True
        discriminator_gradients = disc_tape.gradient(d_loss, disc.trainable_variables)
        disc_opt.apply_gradients(zip(discriminator_gradients, disc.trainable_variables))

    disc.trainable = False
    gan_gradients = gen_tape.gradient(g_loss_total, gan.trainable_variables)
    enc_dec_opt.apply_gradients(zip(gan_gradients, gan.trainable_variables))

    step = enc_dec_opt.iterations
    with sm_writer.as_default():
        tf.summary.scalar('d_loss_perceptual', d_loss, step=step)
        tf.summary.scalar('d_loss_real', d_loss_real, step=step)
        tf.summary.scalar('d_loss_fake', d_loss_fake, step=step)
        tf.summary.scalar('g_loss_perceptual', g_loss, step=step)
        tf.summary.scalar('g_l2_loss', l2_loss, step=step)
        tf.summary.scalar('g_lpips_loss', lpips_loss, step=step)
        tf.summary.scalar('g_feature_matching_loss', lfm_loss, step=step)
        tf.summary.scalar('g_loss_total', g_loss_total, step=step)