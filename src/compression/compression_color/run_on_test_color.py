import sys
import pathlib
import matplotlib.pyplot as plt
import os
import gin
import tensorflow as tf

sys.path.insert(1, os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../../..')))
from src.compression.shared.data_utils import generate_test_image, load_prepare_data_test
from src.compression.shared.net_architecture import make_enc, make_gen, make_gan, make_multi_scale_disc

# os.environ["CUDA_VISIBLE_DEVICES"]="1"


GIN_FIN = '/home/kiadmin/projects/Interactive-Deep-Colorization-and-Compression/src/compression/compression_color/extreme_compression_color.gin'

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
             input_dim_target, data, data_test, test_out, buf_size):
    ckpt_path = base_path + ckpt_dir
    tb_path = base_path + tb_dir
    gen_path = base_path + gen_imgs_dir
    model_path = base_path + model_dir
    log_path = base_path + log_dir
    lpips_path = base_path + lpips_weights
    data = base_path + data
    data_test = base_path + data_test
    test_out = base_path + test_out
    return ckpt_path, tb_path, gen_path, model_path, log_path, input_dim_raw, input_dim_target, data, data_test, test_out, lpips_path, buf_size

gin.parse_config_file(GIN_FIN)
epochs, batch_size, k_beta, k_m, k_p, k_fm, channel_bottleneck, disc_scale = get_shared_specs()
ckpt_path, tb_path, gen_path, model_path, log_path, input_dim_raw, input_dim_target, data, data_test, test_out, lpips_path, buf_size = setup_io()

# load and preprocess testset
#test_ds = load_prepare_data_test(path_to_test)

h_transform, w_transform = input_dim_raw[0] // 16, input_dim_raw[1] // 16
encoder = make_enc(gen_path, num_filters_bottleneck=channel_bottleneck, vis_model=False)
decoder = make_gen((h_transform, w_transform, channel_bottleneck), gen_path, gray=False, vis_model=False)
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
ckpt.restore(tf.train.latest_checkpoint(ckpt_path))

if not os.path.isdir(test_out):
    os.mkdir(test_out)


for filename in os.listdir(data_test)[:50]:
    path_to_file = os.path.join(data_test, filename)
    output_filename = os.path.join(test_out, filename)
    generate_test_image(encoder, decoder, path_to_file, output_filename, gray=False)
