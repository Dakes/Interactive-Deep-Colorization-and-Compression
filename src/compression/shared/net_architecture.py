from src.compression.shared.arch_ops import ConvBlock, ResidualBlock, UpConvBlock, ChannelNorm
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers
import os


def make_enc(gen_path, num_down=4, num_filters_base=60, num_filters_bottleneck=4, vis_model=True, input=(None, None, 3)):
    """
    fully convolutional encoder based on
    - https://arxiv.org/pdf/1804.02958.pdf
    - https://arxiv.org/pdf/1711.11585.pdf
    - https://arxiv.org/pdf/1603.08155.pdf

    implementational deviations based on
    - https://arxiv.org/pdf/2006.09965.pdf and
    - https://github.com/tensorflow/compression/blob/master/models/hific/archs.py

    For the GC, the encoder E convolutionally processes the image x and optionally the label map s, with spatial
    dimension W × H, into a feature map of size W/16 × H/16 × 960 (with 6 layers, of which four have 2-strided
    convolutions), which is then projected down to C channels (where C ∈ {2, 4, 8} is much smaller than 960). This
    results in a feature map w of dimension W/16 × H/16 × C.

    naming conventions:
    - c7s1-k denote a 7 × 7 Convolution-ChannelNorm-ReLU layer with k filters and stride 1
    - dk denotes a 3 × 3 Convolution-ChannelNorm-ReLU layer with k filters, and stride 2.

    TODOs:
    - replace LayerNorm with ChannelNorm according to https://arxiv.org/pdf/2006.09965.pdf  [ok]

    :param gen_path:
    :param num_down:
    :param num_filters_base:
    :param num_filters_bottleneck:
    :param vis_model:
    :return:
    """

    inp_img = layers.Input(shape=input, name='input_image')

    # c7s1-60, d120, d240, d480, d960, c3s1-C
    w = ConvBlock(filters=num_filters_base, kernel_size=(7, 7), strides=(1, 1), name='c7s1-60')(inp_img)
    for i in range(num_down):
        num_filters = num_filters_base * 2 ** (i + 1)
        w = ConvBlock(filters=num_filters, kernel_size=(3, 3), strides=(2, 2), name='d' + str(num_filters))(w)
    # ConvBlock without ChannelNorm and ReLu activation in final layer
    w = layers.Conv2D(filters=num_filters_bottleneck, kernel_size=(3, 3), padding='same', name='c3s1-C')(w)

    # we want our final values to be in range [-2, 2] -> see quantizer for more information
    w = 2 * tf.nn.tanh(w)

    # define model
    model = tf.keras.Model(inp_img, w)
    model.summary()

    if vis_model:
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=gen_path + 'Encoder.png')

    return model


def quantizer_theis(w):
    """
    straight-through estimator based on Theis 2017:
    https://arxiv.org/pdf/1703.00395.pdf

    -> use rounding, pass gradients unchanged
    -> advantage: same quantization procedure for both training and inference

    note: by varying the scaling of tanh in the encoder, different quantization ranges become possible
    e.g. 2 * tf.nn.tanh(w) equals the range [-2, -1, 0, 1, 2] proposed in quantizer_agustsson (see below)

    implementation based on https://github.com/tensorflow/compression/blob/master/models/hific/archs.py

    :param w:
    :return:
    """
    half = tf.constant(.5, dtype=tf.float32)
    outputs = w
    # Rounding latents for the forward pass (straight-through).
    outputs = outputs + tf.stop_gradient(tf.math.floor(outputs + half) - outputs)
    return outputs


def make_gen(input_dim, gen_path, num_up=4, num_filters_base=60, num_residual_blocks=9, gray=False, vis_model=True):
    """
    fully convolutional decoder based on
    - https://arxiv.org/pdf/1804.02958.pdf
    - https://arxiv.org/pdf/1711.11585.pdf
    - https://arxiv.org/pdf/1603.08155.pdf

    implementational deviations based on
    - https://arxiv.org/pdf/2006.09965.pdf and
    - https://github.com/tensorflow/compression/blob/master/models/hific/archs.py

    The generator G projects w_hat up to 960 channels, processes these with 9 residual units at dimension
    W/16 × H/16 × 960, and then mirrors E by convolutionally processing the features back to spatial dimension W × H
    (with transposed convolutions instead of strided ones)

    naming conventions
    - c7s1-k denote a 7 × 7 Convolution-ChannelNorm [53]-ReLU layer with k filters and stride 1
    - Rk denotes a residual block that contains two 3 × 3 conv layers with the same number of filters on both layers.
    - uk denotes a 3 × 3 fractional-strided-Convolution-ChannelNorm-ReLU layer with k filters, and stride 1/2.

    TODOs:
    - optionally concat noise drawn from a fixed prior distribution                         [skip]
    - replace LayerNorm with ChannelNorm according to https://arxiv.org/pdf/2006.09965.pdf  [ok]

    :param z:
    :param out_channels:
    :return:
    """

    _, _, c = input_dim
    enc_img = layers.Input(shape=(None, None, c), name='encoded_image')

    # c3s1-960, R960, R960, R960, R960, R960, R960, R960, R960, R960, u480, u240, u120, u60, c7s1-3
    num_filters = num_filters_base * (2 ** num_up)
    head = ChannelNorm()(enc_img)  # tfa.layers.InstanceNormalization()(enc_img)
    head = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), padding='same', name='c3s1-960')(head)
    head = ChannelNorm()(head)  # tfa.layers.InstanceNormalization()(head)

    # Residual blocks
    w = head
    for i in range(num_residual_blocks):
        w = ResidualBlock(filters=num_filters, kernel_size=(3, 3), name='R' + str(num_filters) + '_' + str(i))(w)

    # Upsampling blocks
    up = w
    up += head  # (skip connection)
    for scale in reversed(range(num_up)):
        num_filters = num_filters_base * (2 ** scale)
        up = UpConvBlock(filters=num_filters, kernel_size=(3, 3), strides=(2, 2), name='u' + str(num_filters))(up)

    # final conv layer
    # note: tf.clip_by_value(nodes_gen.reconstruction, 0, 255.) is used later on to restore the original values range
    if gray:
        restored_image = layers.Conv2D(filters=1, kernel_size=(7, 7), padding='same', name='c7s1-1')(up)
    else:
        restored_image = layers.Conv2D(filters=3, kernel_size=(7, 7), padding='same', name='c7s1-3')(up)
    restored_image = tf.nn.tanh(restored_image)

    # define model
    model = tf.keras.Model(enc_img, restored_image)
    model.summary()

    if vis_model:
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, to_file=gen_path + 'Generator-Decoder.png')

    return model


def make_multi_scale_disc(input_dim_x, input_dim_y, gen_path, scales=3, vis_model=True):
    """
    Build the Discriminator (multi scale) based on

    - https://arxiv.org/pdf/1804.02958.pdf
    - https://arxiv.org/pdf/2006.09965.pdf
    - https://arxiv.org/pdf/1611.07004.pdf
    - https://github.com/tensorflow/compression/blob/master/models/hific/archs.py

    The 70 × 70 discriminator architecture is: C64-C128-C256-C512. After the last layer, a convolution is applied to map
    to a 1-dimensional output, followed by a Sigmoid function. As an exception to the above notation, BatchNorm is not
    applied to the first C64 layer. All ReLUs are leaky, with slope 0.2.

    naming conventions
    - Let Ck denote a Convolution-SpectralNorm-ReLU layer with k filters.

    TODOs:
    - replace BatchNorm with SpectralNorm                                                   [ok]
    - condition D on y by concatenating an upscaled version to the image                    [ok]

    :param input_dim_x: w_hat e.g. 32 × 64 × 4
    :param input_dim_y: original or reconstructed image, e.g. 512 × 1024 × 3
    :param gen_path:
    :param scales:
    :param vis_model:
    :return:
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    h_x, w_x, c_x = input_dim_x
    h_y, w_y, c_y = input_dim_y

    latent = tf.keras.layers.Input(shape=[h_x, w_x, c_x], name='w_hat')
    tar = tf.keras.layers.Input(shape=[h_y, w_y, c_y], name='target_image')

    disc_outputs = []
    disc_activations = []

    # extract features from latent feature map
    latent_f = conv_patch_disc(12, 3, 1, initializer)(latent)

    for i in tf.range(scales):
        # resize to match respective scale
        tar_i = tf.image.resize(tar, [h_y // 2 ** i, w_y // 2 ** i], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        _, h, w, _ = tar_i.get_shape().as_list()

        # condition D on y by concatenating an upscaled version to the image
        latent_f = tf.image.resize(latent_f, [h, w], tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.keras.layers.concatenate([latent_f, tar_i])

        x1 = conv_patch_disc(64, 4, 2, initializer, False)(x)
        x2 = conv_patch_disc(128, 4, 2, initializer)(x1)
        x3 = conv_patch_disc(256, 4, 2, initializer)(x2)
        x4 = conv_patch_disc(512, 4, 1, initializer)(x3)

        # Final 1x1 conv that maps to 1 Channel
        last = tf.keras.layers.Conv2D(1, kernel_size=3, strides=1, padding='same', kernel_initializer=initializer)(x4)

        # Reshape all into batch dimension.
        last = tf.reshape(last, [-1, 1])
        # last = tf.nn.sigmoid(last) -> sigmoid_cross_entropy_with_logits
        disc_outputs.append(last)
        disc_activations.append([x1, x2, x3, x4])

    model = tf.keras.Model(inputs=[latent, tar], outputs=[disc_outputs, disc_activations])
    model.summary()

    if vis_model:
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'MultiScaleDiscriminator.png')

    return model


def conv_patch_disc(filters, size, strides, initializer, apply_norm=True):
    """
    Conv block based on https://arxiv.org/pdf/1611.07004.pdf
    :param filters:
    :param size:
    :param strides:
    :param initializer:
    :param apply_norm:
    :return:
    """
    result = tf.keras.Sequential()

    if apply_norm:
        result.add(
            tfa.layers.SpectralNormalization(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                                                    kernel_initializer=initializer, use_bias=False)))
    else:
        result.add(tf.keras.layers.Conv2D(filters, size, strides=strides, padding='same',
                                          kernel_initializer=initializer, use_bias=False))

    result.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    return result


def make_gan(encoder, decoder, multi_scale_disc, gen_path, vis_model=True):
    """
    Update Generator (Encoder-Decoder) through composite model

    :param encoder:
    :param decoder:
    :param multi_scale_disc:
    :param gen_path:
    :param vis_model:
    :return:
    """
    multi_scale_disc.trainable = False

    # encoder -> quantize -> decode
    z = quantizer_theis(encoder.output)
    x_hat = decoder(z)
    d_gz_outs, d_gz_acts = multi_scale_disc([z, x_hat])

    # define composite model
    model = tf.keras.Model([encoder.input], [x_hat, z, d_gz_outs, d_gz_acts])
    model.summary()

    if vis_model:
        if not os.path.exists(gen_path):
            os.makedirs(gen_path)
        tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True,
                                  to_file=gen_path + 'Encoder-TheisQuantizer-Decoder.png')

    return model