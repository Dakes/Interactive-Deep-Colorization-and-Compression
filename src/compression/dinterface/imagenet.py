import cv2
import os


def convert_to_gan_reading_format_save(input_dir, output_dir, target_size):
    """
    Prerequisite:
    Download leftImg8bit into res/data/leftImg8bit
    link: https://www.cityscapes-dataset.com/downloads/

    Convert Cityscapes dataset to "GAN format"
    (1) get list of all files in directory tree res/data/leftImg8bit/{train/, val/, test/}
    (2) process/ filter images

    Input:
    res/data/leftImg8bit/
        ├── test
            ├── berlin
                ├── ...
            ├── ...
        ├── train
        ├── val

    Output:
    res/data/leftImg8bit-preprocessed/
        ├── test
            ├── img_i.png
            ├── img_j.png
            ├── ...
        ├── train
        ├── val

    (remove intermediate folders, but keep train, valid, test split)

    :param input_dir:       directory containing Cityscapes
    :param output_dir:      output location
    :param target_size:     (height, width, channels)
    :return:
    """

    h, w, _ = target_size

    # create output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # (1) get list of all files in directory tree res/data/leftImg8bit/{train/, valid/, test/}
    listOfFiles = list()
    valid_samples = 0

    for (dirpath, dirnames, filenames) in os.walk(input_dir):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    # (2) process/ filter images
    for idx, file in enumerate(listOfFiles):

        if file.endswith(".png"):
            print(file)
            img_nm = os.path.basename(file)
            img = cv2.imread(os.path.join(input_dir, file))
            resized_img = cv2.resize(img, (w, h))
            cv2.imwrite(os.path.join(output_dir, img_nm), resized_img)
            valid_samples += 1

    print('size of valid images: {}'.format(valid_samples))