
from vae import *
from PIL import Image
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Flatten, Conv2DTranspose, LeakyReLU
import pickle
from tensorflow import Tensor
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Add

verbose = False


def load_dataset(save_path):
    # load the dataset

    print('loading tensor_spec from pkl file')
    tensor_spec = pickle.load(open(save_path + "tensor_spec.pkl", "rb"))

    print('loading dataset from files.')
    loaded_dataset = tf.data.experimental.load(path=save_path, element_spec=tensor_spec)
    #loaded_dataset = tf.data.experimental.load(path=save_path, compression='GZIP', element_spec=tensor_spec)

    # print('caching dataset to file ')
    # loaded_dataset = loaded_dataset.cache(save_path + 'cache_file.tf')

    print('loaded dataset')
    print(type(loaded_dataset))

    return loaded_dataset


# class_mode='input' is for autoencoders
def get_datagenerator(path, batch_size):
    # create a data generator
    print('loading data...')
    datagen = ImageDataGenerator(rescale=(1 / 255))
    train_generator = datagen.flow_from_directory(path, batch_size=batch_size, shuffle=False, class_mode=None, target_size=(64, 64))
    return train_generator

def concat_four_h(im1, im2, im3, im4):
    ## concatenates four images horizontally
    a = get_concat_h(im1, im2)
    b = get_concat_h(im3, im4)
    result = get_concat_h(a, b)
    return result

def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst

def get_concat_v(im1, im2):
    dst = Image.new('RGB', (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst

def log(text):
    #todo: make this write to a log file
    if verbose:
        print(text)

