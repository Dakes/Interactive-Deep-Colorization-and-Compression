import os, sys
import matplotlib.pyplot as plt
import time
import pathlib
import tensorflow as tf
import numpy as np

sys.path.insert(1, os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../..')))
print(os.path.abspath(os.path.join(pathlib.Path(__file__).parent.resolve(), '../..')))
import model
import input_data
from src.utils.logger import Logger

# from utils import add_color_pixels_rand_gt, imsegkmeans
from src.utils import files

dirs = files.config_parse(dirs=True)


# TODO: remove. These are technically not needed any more, paths are directly set in preprocess from config
_BASE_PATH = '/home/kiadmin/projects/Interactive-Deep-Colorization-and-Compression/'

_IMAGE_COLOR_DIR = dirs["train"] + dirs["ground_truth"]
_COLOR_MAP_DIR = dirs["train"] + dirs["color_map"]
_THEME_DIR = dirs["train"] + dirs["theme_rgb"]
_THEME_MASK_DIR = dirs["train"] + dirs["theme_mask"]
_LOCAL_DIR = dirs["train"] + dirs["local_hints"]
_LOCAL_MASK_DIR = dirs["train"] + dirs["local_mask"]

_EVAL_IMG_RGB = dirs["val"] + dirs["ground_truth"]
_EVAL_THEME_RGB = dirs["val"] + dirs["theme_rgb"]
_EVAL_MASK = dirs["val"] + dirs["theme_mask"]
_EVAL_POINTS_RGB = dirs["val"] + dirs["local_hints"]
_EVAL_POINTS_MASK = dirs["val"] + dirs["local_mask"]

# _LOGS_DIR = _BASE_PATH + '/res/logs/'
_LOGS_DIR = dirs["colorization_train"]
_EXT_LIST = ['png', 'png', 'png', 'png', 'png', 'png']
_NAME_LIST = ['gt img', 'theme img', 'theme mask', 'color_map img', 'local img', 'local mask']


# first step: without residual network
def train1():
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 300001
    IMAGE_SIZE = 256

    # the parameters to differentiate the influences of three parts
    l1 = 0.9
    l2 = 0.1

    sess = tf.compat.v1.Session()

    # directory of checkpoint
    logs_dir = _LOGS_DIR + 'run_1/'
    train_logger = Logger()
    test_logger = Logger()
    fw = tf.compat.v1.summary.FileWriter(logs_dir, graph=sess.graph)
    # fw = tf.summary.create_file_writer(logs_dir)

    # get the training data
    train_list = input_data.get_train_list(
        [_IMAGE_COLOR_DIR, _THEME_DIR, _THEME_MASK_DIR, _COLOR_MAP_DIR, _LOCAL_DIR, _LOCAL_MASK_DIR],
        _NAME_LIST, _EXT_LIST, shuffle=True)

    image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
        input_data.get_batch(train_list, (IMAGE_SIZE, IMAGE_SIZE), BATCH_SIZE, CAPACITY, True)

    image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
    image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_gra_batch = model.sobel(image_l_batch)
    image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    index_lab_batch = input_data.rgb_to_lab(index_rgb_batch)
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255.0 * 2 - 1

    point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
    point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    # TODO: training
    out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch,
                                      point_ab_batch, point_mask_batch,
                                      is_training=True, scope_name='UserGuide')

    # TODO: evaluate
    image_l_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test, image_rgb_test, image_gra_test = \
        input_data.get_eval_img(_EVAL_IMG_RGB, _EVAL_THEME_RGB, _EVAL_MASK, _EVAL_POINTS_RGB, _EVAL_POINTS_MASK)
    test_ab_out = model.inference3_1(image_l_test, image_gra_test, theme_ab_test, theme_mask_test, point_ab_test,
                                     point_mask_test, is_training=False, scope_name='UserGuide')
    test_rgb_out = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out + 1.) / 2 * 255. - 128], axis=3))
    test_psnr = 10 * tf.math.log(1 / (tf.reduce_mean(input_tensor=tf.square(test_rgb_out - image_rgb_test)))) / np.log(10)

    global_step = tf.compat.v1.train.get_or_create_global_step(sess.graph)
    # compute the loss
    train_loss, loss_paras = model.loss_colorization(out_ab_batch, image_ab_batch, index_ab_batch, l1, l2)

    # added by @Nikolai10
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_col_total', train_loss)])
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_col_1', loss_paras[0])])
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_col_2', loss_paras[1])])
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_col_3', loss_paras[2])])
    train_logger.finalize_with_sess(sess)

    test_logger.add_summaries([tf.compat.v1.summary.scalar('test/psnr', test_psnr),
                               tf.compat.v1.summary.image('test/rgb_out', test_rgb_out)])
    test_logger.finalize_with_sess(sess)

    var_list = tf.compat.v1.trainable_variables()
    paras_count = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in var_list])
    print('#Params:%d' % sess.run(paras_count), end='\n\n')

    train_op, learning_rate = model.training(train_loss, global_step, 1e-3, 4e4, 0.7, var_list)

    saver1 = tf.compat.v1.train.Saver(max_to_keep=10)

    sess.run(tf.compat.v1.global_variables_initializer())  # Variable initialization

    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                print('???')
                break

            _, loss, loss_sub, lr = sess.run([train_op, train_loss, loss_paras, learning_rate])

            if step % 100 == 0:
                train_logger.log().to_tensorboard(fw, step)
                runtime = time.time() - s_t
                psnr = sess.run(test_psnr)
                # record the training process
                print('Step: %d, Loss_total: %g, loss1: %g, test_psnr: %.2fdB, learning_rate:%g, '
                      'time:%.2fs, time left: %.2fhours'
                      % (step, loss, loss_sub[0], psnr, lr,
                         runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 10000 == 0 or step == MAX_STEP - 1:  # save checkpoint
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver1.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0 or step == MAX_STEP - 1:
                test_logger.log().to_tensorboard(fw, step)
                test_out = sess.run(test_rgb_out)
                test_out = test_out[0]

                save_path = logs_dir + 'images/'  # save results
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path += 'step_' + str(step) + '.png'
                plt.imsave(save_path, test_out)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    # Wait for the thread to end
    coord.join(threads=threads)
    sess.close()


# second step: fix the colorization network and train the residual network
def train2():
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 300001
    IMAGE_SIZE = 256

    sess = tf.compat.v1.Session()

    # directory of checkpoint
    logs_ckpts = _LOGS_DIR + 'run_1/'
    logs_dir = _LOGS_DIR + 'run_2/'
    train_logger = Logger()
    test_logger = Logger()
    # fw = tf.compat.v1.summary.FileWriter(logs_dir, graph=sess.graph)
    fw = tf.compat.v1.summary.FileWriter(logs_dir, graph=sess.graph)
    # fw = tf.summary.create_file_writer(logs_dir)

    # get the training data
    train_list = input_data.get_train_list(
        [_IMAGE_COLOR_DIR, _THEME_DIR, _THEME_MASK_DIR, _COLOR_MAP_DIR, _LOCAL_DIR, _LOCAL_MASK_DIR],
        _NAME_LIST, _EXT_LIST, shuffle=True)

    image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
        input_data.get_batch(train_list, (IMAGE_SIZE, IMAGE_SIZE), BATCH_SIZE, CAPACITY, True)

    image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
    image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_gra_batch = model.sobel(image_l_batch)
    image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    index_lab_batch = input_data.rgb_to_lab(index_rgb_batch)
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255.0 * 2 - 1

    point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
    point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    # TODO: training
    # colorization network
    out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch,
                                      point_ab_batch, point_mask_batch,
                                      is_training=False, scope_name='UserGuide')
    # residual network
    _, out_ab_batch2 = model.gen_PRLNet(out_ab_batch, image_l_batch, 2, scope_name='PRLNet')

    # TODO: evaluate
    image_l_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test, image_rgb_test, image_gra_test = \
        input_data.get_eval_img(_EVAL_IMG_RGB, _EVAL_THEME_RGB, _EVAL_MASK, _EVAL_POINTS_RGB, _EVAL_POINTS_MASK)
    test_ab_out = model.inference3_1(image_l_test, image_gra_test, theme_ab_test, theme_mask_test, point_ab_test,
                                     point_mask_test,
                                     is_training=False, scope_name='UserGuide')
    _, test_ab_out2 = model.gen_PRLNet(test_ab_out, image_l_test, 2, scope_name='PRLNet')
    test_rgb_out0 = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out + 1.) / 2 * 255. - 128], axis=3))
    test_rgb_out = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out2 + 1.) / 2 * 255. - 128], axis=3))
    test_psnr0 = 10 * tf.math.log(1 / (tf.reduce_mean(input_tensor=tf.square(test_rgb_out0 - image_rgb_test)))) / np.log(10)
    test_psnr = 10 * tf.math.log(1 / (tf.reduce_mean(input_tensor=tf.square(test_rgb_out - image_rgb_test)))) / np.log(10)

    # 训练残差网络 / Train the residual network
    var_list = tf.compat.v1.global_variables()
    var_model1 = [var for var in var_list if var.name.startswith('UserGuide')]
    var_model2 = [var for var in var_list if var.name.startswith('PRLNet')]

    paras_count1 = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in var_model1])
    paras_count2 = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in var_model2])
    print('UserGuide参数数目:%d' % sess.run(paras_count1))
    print('Detailed参数数目:%d' % sess.run(paras_count2))

    global_step = tf.compat.v1.train.get_or_create_global_step(sess.graph)
    train_loss, loss_paras = model.loss_residual(out_ab_batch2, image_ab_batch)
    train_op, learning_rate = model.training(train_loss, global_step, 1e-4, 4e4, 0.7, var_model2)

    # added by @Nikolai10
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_res_total', train_loss)])
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_res_1', loss_paras[0])])
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_res_2', loss_paras[1])])
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_res_3', loss_paras[2])])
    train_logger.finalize_with_sess(sess)

    test_logger.add_summaries([tf.compat.v1.summary.scalar('test/psnr0', test_psnr0),
                               tf.compat.v1.summary.scalar('test/psnr', test_psnr),
                               tf.compat.v1.summary.image('test/rgb_out0', test_rgb_out0),
                               tf.compat.v1.summary.image('test/rgb_out', test_rgb_out)])
    test_logger.finalize_with_sess(sess)

    saver1 = tf.compat.v1.train.Saver(var_list=var_model1)
    saver2 = tf.compat.v1.train.Saver(var_list=var_model2)

    sess.run(tf.compat.v1.global_variables_initializer())

    print('Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(logs_ckpts)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver1.restore(sess, ckpt.model_checkpoint_path)
        print('Success, global_step = %s' % global_step)
    else:
        print('Fail')

    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                print('???')
                break

            _, loss, loss_sub, lr = sess.run([train_op, train_loss, loss_paras, learning_rate])

            if step % 100 == 0:
                train_logger.log().to_tensorboard(fw, step)
                runtime = time.time() - s_t
                psnr0, psnr = sess.run([test_psnr0, test_psnr])
                print('Step: %d, Loss_total: %g, loss1: %g, test_psnr0: %.2fdB, test_psnr: %.2fdB, learning_rate:%g, '
                      'time:%.2fs, time left: %.2fhours'
                      % (step, loss, loss_sub[0], psnr0, psnr, lr,
                         runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 10000 == 0 or step == MAX_STEP - 1:
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver2.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0 or step == MAX_STEP - 1:
                test_logger.log().to_tensorboard(fw, step)
                test_out = sess.run(test_rgb_out)
                test_out = test_out[0]

                save_path = logs_dir + 'images/'  # save results
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path += 'step_' + str(step) + '.png'
                plt.imsave(save_path, test_out)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    # Wait for the thread to end
    coord.join(threads=threads)
    sess.close()


# jointly train the two networks
def train3():
    BATCH_SIZE = 16
    CAPACITY = 1000
    MAX_STEP = 240001
    IMAGE_SIZE = 256

    # the parameters to differentiate the influences of three parts
    l1 = 0.9
    l2 = 0.1

    sess = tf.compat.v1.Session()

    # directory of checkpoint
    logs_ckpts_run_1 = _LOGS_DIR + 'run_1/'
    logs_ckpts_run_2 = _LOGS_DIR + 'run_2/'
    logs_dir = _LOGS_DIR + 'run_3/'
    train_logger = Logger()
    test_logger = Logger()
    fw = tf.compat.v1.summary.FileWriter(logs_dir, graph=sess.graph)

    # get the training data
    train_list = input_data.get_train_list(
        [_IMAGE_COLOR_DIR, _THEME_DIR, _THEME_MASK_DIR, _COLOR_MAP_DIR, _LOCAL_DIR, _LOCAL_MASK_DIR],
        _NAME_LIST, _EXT_LIST, shuffle=True)

    image_rgb_batch, theme_rgb_batch, theme_mask_batch, index_rgb_batch, point_rgb_batch, point_mask_batch = \
        input_data.get_batch(train_list, (IMAGE_SIZE, IMAGE_SIZE), BATCH_SIZE, CAPACITY, True)

    image_lab_batch = input_data.rgb_to_lab(image_rgb_batch)
    image_l_batch = tf.reshape(image_lab_batch[:, :, :, 0] / 100.0 * 2 - 1, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 1])
    image_l_gra_batch = model.sobel(image_l_batch)
    image_ab_batch = (image_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    theme_lab_batch = input_data.rgb_to_lab(theme_rgb_batch)
    theme_ab_batch = (theme_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    index_lab_batch = input_data.rgb_to_lab(index_rgb_batch)
    index_ab_batch = (index_lab_batch[:, :, :, 1:] + 128) / 255.0 * 2 - 1

    point_lab_batch = input_data.rgb_to_lab(point_rgb_batch)
    point_ab_batch = (point_lab_batch[:, :, :, 1:] + 128.) / 255.0 * 2 - 1

    # TODO: training
    out_ab_batch = model.inference3_1(image_l_batch, image_l_gra_batch, theme_ab_batch, theme_mask_batch,
                                      point_ab_batch, point_mask_batch,
                                      is_training=True, scope_name='UserGuide')
    _, out_ab_batch2 = model.gen_PRLNet(out_ab_batch, image_l_batch, 2, scope_name='PRLNet')

    # TODO: envalute
    image_l_test, theme_ab_test, theme_mask_test, point_ab_test, point_mask_test, image_rgb_test, image_gra_test = \
        input_data.get_eval_img(_EVAL_IMG_RGB, _EVAL_THEME_RGB, _EVAL_MASK, _EVAL_POINTS_RGB, _EVAL_POINTS_MASK)
    test_ab_out = model.inference3_1(image_l_test, image_gra_test, theme_ab_test, theme_mask_test, point_ab_test,
                                     point_mask_test,
                                     is_training=False, scope_name='UserGuide')
    _, test_ab_out2 = model.gen_PRLNet(test_ab_out, image_l_test, 2, scope_name='PRLNet')
    test_rgb_out0 = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out + 1.) / 2 * 255. - 128], axis=3))
    test_rgb_out = \
        input_data.lab_to_rgb(tf.concat([(image_l_test + 1.) / 2 * 100., (test_ab_out2 + 1.) / 2 * 255. - 128], axis=3))
    test_psnr0 = 10 * tf.math.log(1 / (tf.reduce_mean(input_tensor=tf.square(test_rgb_out0 - image_rgb_test)))) / np.log(10)
    test_psnr = 10 * tf.math.log(1 / (tf.reduce_mean(input_tensor=tf.square(test_rgb_out - image_rgb_test)))) / np.log(10)

    var_list = tf.compat.v1.global_variables()
    var_model1 = [var for var in var_list if var.name.startswith('UserGuide')]
    var_model2 = [var for var in var_list if var.name.startswith('PRLNet')]
    var_total = var_model1 + var_model2
    paras_count1 = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in var_model1])
    paras_count2 = tf.reduce_sum(input_tensor=[tf.reduce_prod(input_tensor=v.shape) for v in var_model2])
    print('UserGuide参数数目:%d' % sess.run(paras_count1))
    print('Detailed参数数目:%d' % sess.run(paras_count2))

    global_step = tf.compat.v1.train.get_or_create_global_step(sess.graph)
    # the loss function
    # potential error!
    loss1, _ = model.loss_colorization(out_ab_batch, image_ab_batch, index_ab_batch, l1, l2)
    loss2, _ = model.loss_residual(out_ab_batch2, image_ab_batch)
    total_loss = loss1 + loss2
    train_op, learning_rate = model.training(total_loss, global_step, 1e-3, 4e4, 0.7, var_total)

    # added by @Nikolai10
    train_logger.add_summaries([tf.compat.v1.summary.scalar('train/loss_total', total_loss),
                                tf.compat.v1.summary.scalar('train/loss_col_total', loss1),
                                tf.compat.v1.summary.scalar('train/loss_res_total', loss2)])
    train_logger.finalize_with_sess(sess)
    test_logger.add_summaries([tf.compat.v1.summary.scalar('test/psnr0', test_psnr0),
                               tf.compat.v1.summary.scalar('test/psnr', test_psnr),
                               tf.compat.v1.summary.image('test/rgb_out0', test_rgb_out0),
                               tf.compat.v1.summary.image('test/rgb_out', test_rgb_out)])
    test_logger.finalize_with_sess(sess)

    saver1 = tf.compat.v1.train.Saver(var_list=var_model1)
    saver2 = tf.compat.v1.train.Saver(var_list=var_model2)
    saver3 = tf.compat.v1.train.Saver(var_list=var_total)

    sess.run(tf.compat.v1.global_variables_initializer())  # 变量初始化
    print('Loading checkpoint...')
    ckpt = tf.train.get_checkpoint_state(logs_ckpts_run_1)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver1.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    ckpt = tf.train.get_checkpoint_state(logs_ckpts_run_2)
    if ckpt and ckpt.model_checkpoint_path:
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        saver2.restore(sess, ckpt.model_checkpoint_path)
        print('载入成功, global_step = %s' % global_step)
    else:
        print('载入失败')

    coord = tf.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners(sess=sess, coord=coord)

    s_t = time.time()
    try:
        for step in range(MAX_STEP):
            if coord.should_stop():
                print('???')
                break

            _, loss, lr = sess.run([train_op, total_loss, learning_rate])

            if step % 100 == 0:
                train_logger.log().to_tensorboard(fw, step)
                runtime = time.time() - s_t
                psnr0, psnr = sess.run([test_psnr0, test_psnr])
                print('Step: %d, Loss_total: %g, test_psnr0: %.2fdB, test_psnr: %.2fdB, learning_rate:%g, '
                      'time:%.2fs, time left: %.2fhours'
                      % (step, loss, psnr0, psnr, lr, runtime, (MAX_STEP - step) * runtime / 360000))
                s_t = time.time()

            if step % 10000 == 0 or step == MAX_STEP - 1:
                if not os.path.exists(logs_dir):
                    os.makedirs(logs_dir)
                checkpoint_path = os.path.join(logs_dir, 'model.ckpt')
                saver3.save(sess, checkpoint_path, global_step=step)

            if step % 1000 == 0 or step == MAX_STEP - 1:
                test_logger.log().to_tensorboard(fw, step)
                test_out = sess.run(test_rgb_out)
                test_out = test_out[0]

                save_path = logs_dir + 'images/'  # save results
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                save_path += 'step_' + str(step) + '.png'
                plt.imsave(save_path, test_out)

    except tf.errors.OutOfRangeError:
        print('Done.')
    finally:
        coord.request_stop()

    # Wait for the thread to end
    coord.join(threads=threads)
    sess.close()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    tf.compat.v1.disable_eager_execution()
    # train1()
    # train2()
    train3()

