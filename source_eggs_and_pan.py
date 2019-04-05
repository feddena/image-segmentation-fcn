import random
import cv2
import os

import numpy as np

from collections import namedtuple

Label = namedtuple('Label', ['name', 'color'])


def rgb2bgr(tpl):
    return [tpl[2], tpl[1], tpl[0]]


label_defs = [
    Label('unlabeled', rgb2bgr((0, 0, 0))),
    Label('egg', rgb2bgr((128, 128, 128))),
    Label('pan', rgb2bgr((255, 255, 255)))]


def build_file_list(path_to_folder):
    images_filenames = os.listdir(path_to_folder + '/images/')
    file_list = []
    for image_name in images_filenames:
        file_list.append((path_to_folder + "/images/" + image_name, path_to_folder + "/masks/" + image_name))
    return file_list


def image_resize(image, width, height, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def resize_with_padding(image, size):
    h, w = image.shape[:2]
    if h/size[0] == w/size[1]:
        return image_resize(image, None, size[0])

    if h/size[0] > w/size[1]:
        resized = image_resize(image, None, size[1])
    else:
        resized = image_resize(image, size[0], None)

    h_res, w_res = resized.shape[:2]

    print(resized.shape)

    return cv2.copyMakeBorder(resized, 0, min(0, size[0] - h_res), 0, min(0,size[1] - w_res), cv2.BORDER_CONSTANT, value=[0, 0, 0])


class EggsAndPansSource:
    def __init__(self):
        self.image_size = (1280, 720)
        self.num_classes = len(label_defs)

        self.label_colors = {i: np.array(l.color) for i, l \
                             in enumerate(label_defs)}

        self.num_training = None
        self.num_validation = None
        self.train_generator = None
        self.valid_generator = None

    def load_data(self, data_dir, valid_fraction):
        """
        Load the data and make the generators
        :param data_dir:       the directory where the dataset's file are stored
        :param valid_fraction: what franction of the dataset should be used
                               as a validation sample
        """
        train_root = data_dir + '/dataset_formatted/train'
        validation_root = data_dir + '/dataset_formatted/validation'

        train_images = build_file_list(train_root)
        valid_images = build_file_list(validation_root)

        if len(train_images) == 0:
            raise RuntimeError('No training images found in ' + data_dir)
        if len(valid_images) == 0:
            raise RuntimeError('No validatoin images found in ' + data_dir)

        self.num_training = len(train_images)
        self.num_validation = len(valid_images)
        self.train_generator = self.batch_generator(train_images)
        self.valid_generator = self.batch_generator(valid_images)

    def batch_generator(self, image_paths):
        def gen_batch(batch_size, names=False):
            random.shuffle(image_paths)
            for offset in range(0, len(image_paths), batch_size):
                files = image_paths[offset:offset + batch_size]

                images = []
                labels = []
                names_images = []
                names_labels = []
                for f in files:
                    image_file = f[0]
                    label_file = f[1]

                    image = cv2.imread(image_file)
                    label = cv2.imread(label_file)

                    if image is None or label is None:
                        continue

                    image = resize_with_padding(image, self.image_size)
                    label = resize_with_padding(label, self.image_size)

                    label_bg = np.zeros([image.shape[0], image.shape[1]], dtype=bool)
                    label_list = []
                    for ldef in label_defs[1:]:
                        label_current = np.all(label == ldef.color, axis=2)
                        label_bg |= label_current
                        label_list.append(label_current)

                    label_bg = ~label_bg
                    label_all = np.dstack([label_bg, *label_list])
                    label_all = label_all.astype(np.float32)

                    images.append(image.astype(np.float32))
                    labels.append(label_all)

                    if names:
                        names_images.append(image_file)
                        names_labels.append(label_file)

                if names:
                    yield np.array(images), np.array(labels), \
                          names_images, names_labels
                else:
                    yield np.array(images), np.array(labels)

        return gen_batch


def get_source():
    return EggsAndPansSource()
