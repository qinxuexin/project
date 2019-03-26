import cv2
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from generator import train_preprocessing, valid_preprocessing


def getDataGenerator(is_train):
    if is_train:
        data_g = ImageDataGenerator(
            rotation_range=20.,
            # width_shift_range=0.1,
            # height_shift_range=0.1,
            # shear_range=0.1,
            # zoom_range=0.1,
            # channel_shift_range=30,
            fill_mode='nearest',
            horizontal_flip=True,
            vertical_flip=True,
            rescale=1. / 255
        )
    else:
        data_g = ImageDataGenerator(
            rescale=1. / 255
        )
    return data_g


def generator(datain, class_num, batch_size=32, is_train=True):
    cate = 'train' if is_train else 'valid'
    files = datain['train'].keys() if is_train else datain['valid'].keys()
    files = list(files)
    while 1:
        cnt = 0
        x = []
        y = []
        np.random.shuffle(files)
        for f in files:
        #     dir_idx = 0
        #     if len(f.split('_')) == 2:
        #         dir_idx = 1
            img = cv2.imread(f)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if cate == 'train':
                img = train_preprocessing(img)
            else:
                img = valid_preprocessing(img)
            # img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
            label = datain[cate][f]
            x.append(img)
            y.append(label)
            cnt += 1
            if cnt % batch_size == 0:
                x = np.array(x)
                y = np.array(y)
                y = to_categorical(y, class_num)
                # gen = get_DataGenerator(is_train).flow(x, y, batch_size, shuffle=False)
                # g = next(gen)
                # yield g
                yield  x, y
                x = []
                y = []
                cnt = 0

