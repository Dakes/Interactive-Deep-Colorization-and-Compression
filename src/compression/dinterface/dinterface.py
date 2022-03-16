import src.dinterface.imagenet as imagenet


def init_reading(input, output, target_size):
    """
    Convert Dataset (leftImg8bit) dataset to "GAN format"

    :param input:
    :param output:
    :param target_size:
    :return:

    """
    if 'leftImg8bit' in input:
        # convert imagenet Dataset
        imagenet.convert_to_gan_reading_format_save(input + 'train/', output + 'train/', target_size)
        imagenet.convert_to_gan_reading_format_save(input + 'val/', output + 'val/', target_size)
        imagenet.convert_to_gan_reading_format_save(input + 'test/', output + 'test/', target_size)
    else:
        raise ValueError('dataset not supported, please have a look at src.dinterface for more information')