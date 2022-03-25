import sys
import pathlib
import matplotlib.pyplot as plt
import datetime
import os
import gin
import tensorflow as tf

sys.path.insert(1, os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../../..')))
from src.compression.shared.data_utils import load_prepare_data_val, load_prepare_data_train, fit
from src.compression.shared.net_architecture import make_enc, make_gen, make_gan, make_multi_scale_disc

os.environ["CUDA_VISIBLE_DEVICES"]="1"

GIN_FIN = 'src/compression/compression_grayscale/extreme_compression_gray.gin'
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
def setup_io(base_path, ckpt_dir, tb_dir, gen_imgs_dir, model_dir, log_dir, lpips_weights, input_dim_raw,
             input_dim_target, data, data_prep, buf_size):
    ckpt_path = base_path + ckpt_dir
    tb_path = base_path + tb_dir
    gen_path = base_path + gen_imgs_dir
    model_path = base_path + model_dir
    log_path = base_path + log_dir
    lpips_path = base_path + lpips_weights
    data = base_path + data
    data_prep = base_path + data_prep
    return ckpt_path, tb_path, gen_path, model_path, log_path, input_dim_raw, input_dim_target, data, data_prep, lpips_path, buf_size


gin.parse_config_file(GIN_FIN)
epochs, batch_size, k_beta, k_m, k_p, k_fm, channel_bottleneck, disc_scale = get_shared_specs()
ckpt_path, tb_path, gen_path, model_path, log_path, input_dim_raw, input_dim_target, data, data_prep, lpips_path, buf_size = setup_io()

print('Experimental setting \nnumber of training epochs: {}\nbatch size: {}\nbpp: {}'.format(epochs, batch_size, 0.072 if channel_bottleneck == 8 else '?'))

"""
# convert imagenet dataset to GAN format
if not os.path.exists(data_prep):
    print('converting to GAN format...')
    init_reading(data, data_prep, input_dim_raw)
"""
# preprocess_color("/home/daniel/imagenet-mini", -1, -1)



# load and preprocess dataset
train_ds = load_prepare_data_train(data_prep, batch_size, buf_size, True)
val_ds = load_prepare_data_val(data_prep, True)

h_transform, w_transform = input_dim_raw[0] // 16, input_dim_raw[1] // 16
encoder = make_enc(gen_path, num_filters_bottleneck=channel_bottleneck, vis_model=False, input=(None, None, 1))
decoder = make_gen((h_transform, w_transform, channel_bottleneck), gen_path, gray=True, vis_model=False)
disc = make_multi_scale_disc((h_transform, w_transform, channel_bottleneck), input_dim_raw, gen_path, vis_model=False)

# build composite model (update G through composite model)
gan = make_gan(encoder, decoder, disc, gen_path, vis_model=False)

enc_dec_opt, disc_opt, gan_loss, use_lpips, use_feature_matching = setup_optimizer()
ckpt_pref = os.path.join(ckpt_path, "ckpt")
ckpt = tf.train.Checkpoint(encoder_decoder_optimizer=enc_dec_opt,
                               discriminator_optimizer=disc_opt,
                               discriminator=disc,
                               encoder=encoder,
                               decoder=decoder)

sum_wr = tf.summary.create_file_writer(log_path + "fit/" + datetime.datetime.now().strftime(STRFTIME_FORMAT))
fit(train_ds, val_ds, epochs, ckpt, ckpt_pref, gan, encoder, decoder, disc, sum_wr, enc_dec_opt, disc_opt, gan_loss, k_beta, k_m, k_p,
        k_fm, gen_path, lpips_path, model_path, use_lpips, use_feature_matching, gray=True, warm_up=True)

ckpt.restore(tf.train.latest_checkpoint(ckpt_path))


# Run the trained model on a few examples from the test dataset
#for inp in val_ds.take(25):
#    generate_images(encoder, decoder, inp)

#fit(train_ds, val_ds, epochs, ckpt, ckpt_pref, gan, encoder, decoder, disc, sum_wr, enc_dec_opt, disc_opt, gan_loss, k_beta, k_m, k_p,
#        k_fm, gen_path, lpips_path, model_path, use_lpips, use_feature_matching, gray=True warm_up=False)


# Run the trained model on a few examples from the test dataset
#for inp in val_ds.take(25):
#    generate_images(encoder, decoder, inp)

