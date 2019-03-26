import numpy as np
import cv2
from keras.preprocessing.image import DirectoryIterator
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet50 import preprocess_input
np.random.seed(3)


def resize_image(
        x,
        size_ideal=(256, 256),
        scale_rate=1.0,
        keep_aspect=False,
        random_scale=False):
    # transform input x to array

    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # coefficients for resize
    if len(img.shape) == 4:
        f, img_h, img_w, img_l, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        img_h, img_w, img_l, = img.shape

    if len(size_ideal) == 1:
        ideal_h = size_ideal
        ideal_w = size_ideal
    if len(size_ideal) == 2:
        ideal_h = size_ideal[0]
        ideal_w = size_ideal[1]
    if size_ideal is None:
        ideal_h = img_h * scale_rate
        ideal_w = img_w * scale_rate

    coef_h,coef_w = 1, 1
    if img_h < ideal_h:
        coef_h = ideal_h / img_h
    if img_w < ideal_w:
        coef_w = ideal_w / img_w

    # Calculate coef to match low size to ideal one

    low_scale = scale_rate
    if random_scale:
        low_scale = 1.0
    coef_max = max(coef_h, coef_w) * np.random.uniform(low=low_scale, high=scale_rate)

    # Resize image
    resize_h = np.ceil(img_h * coef_max)
    resize_w = np.ceil(img_w * coef_max)

    method_interpolation = cv2.INTER_CUBIC

    if keep_aspect:
        resize_img = cv2.resize(
            img,
            dsize=(int(resize_w), int(resize_h)),
            interpolation=method_interpolation)
    else:
        resize_img = cv2.resize(
            img,
            dsize=(
                int(ideal_w * np.random.uniform(low=low_scale, high=scale_rate)),
                int(ideal_h * np.random.uniform(low=low_scale, high=scale_rate))),
            interpolation=method_interpolation)

    return resize_img


def center_crop_image(x, size_ideal=(224, 224)):

    # Convert input x to array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Set size
    if len(size_ideal) == 1:
        ideal_h = size_ideal
        ideal_w = size_ideal
    if len(size_ideal) == 2:
        ideal_h = size_ideal[0]
        ideal_w = size_ideal[1]

    if len(img.shape) == 4:
        f, img_h, img_w, img_l, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        img_h, img_w, img_l, = img.shape

    # Crop image
    h_initial = int((img_h - ideal_h) / 2)
    w_initial = int((img_w - ideal_w) / 2)
    img_crop = img[h_initial:h_initial + ideal_h, w_initial:w_initial + ideal_w, :]

    return img_crop


def random_crop_image(x,
                      size_ideal=(224, 224)):

    # Convert input x to array
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x

    # Set size
    if len(size_ideal) == 1:
        ideal_h = size_ideal
        ideal_w = size_ideal
    if len(size_ideal) == 2:
        ideal_h = size_ideal[0]
        ideal_w = size_ideal[1]

    if len(img.shape) == 4:
        f, img_h, img_w, img_l, = img.shape
        img = img[0]
    elif len(img.shape) == 3:
        img_h, img_w, img_l, = img.shape

    # Crop image
    r_h = abs(img_h - ideal_h)
    r_w = abs(img_w - ideal_w)
    h_initial = 0
    w_initial = 0
    if r_h != 0:
        h_initial = np.random.randint(low=0, high=r_h)
    if r_w != 0:
        w_initial = np.random.randint(low=0, high=r_w)
    img_cropped = img[h_initial:h_initial + ideal_h, w_initial:w_initial + ideal_w, :]

    return img_cropped


def horizontal_flip_image(x):

    if np.random.random() >= 0.5:
        return x[:, ::-1, :]
    else:
        return x


def pre(x):
    if not isinstance(x, np.ndarray):
        img = np.asarray(x)
    else:
        img = x
    img = np.expand_dims(img, axis=0)
    y = preprocess_input(img)
    return y[0]


# def normalize_image(x, mean=(0., 0., 0.), std=(1.0, 1.0, 1.0)):
#   x = np.asarray(x, dtype=np.float32)
#    if len(x.shape) == 4:
#         for d in range(3):
#            x[:, :, :, d] = (x[:, :, :, d] - mean[d]) / std[d]
#    if len(x.shape) == 3:
#         for d in range(3):
#            x[:, :, d] = (x[:, :, d] - mean[d]) / std[d]
#
#     return x


# def preprocess_input(x):

#    return normalize_image(x, mean=[123.82988033, 127.3509729, 110.25606303])


class ImageDataGenerator(ImageDataGenerator):
    '''Inherit from keras' ImageDataGenerator.'''

    def flow_from_directory(
            self, directory,
            target_size=(112, 112), color_mode='rgb',
            classes=None, class_mode='categorical',
            batch_size=16, shuffle=True, seed=None,
            save_to_dir=None,
            save_prefix='',
            save_format='png',
            follow_links=False,
            subset=None,
            interpolation='nearest'
    ):
        return DirectoryIterator(
            directory, self,
            target_size=target_size, color_mode=color_mode,
            classes=classes, class_mode=class_mode,
            data_format=self.data_format,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            follow_links=follow_links,
            subset=subset,
            interpolation=interpolation)


def train_preprocessing(x, size_target=(224, 224)):
    return pre(
        random_crop_image(
            horizontal_flip_image(
                resize_image(
                    x,
                    size_ideal=size_target,
                    keep_aspect=True
                )
            )
        )
    )


def valid_preprocessing(x):
    return pre(x)


def get_DataGenerator(is_train):
    if is_train:
        datagen = ImageDataGenerator(
            preprocessing_function=train_preprocessing
        )
    else:
        datagen = ImageDataGenerator(
            preprocessing_function=valid_preprocessing
        )
    return datagen


if __name__ == "__main__":
    pass
