
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



def do_inference(model, epoch):

    #todo: this is aweful

    if epoch is None:
        epoch = 'init'
    else:
        epoch = epoch + 1

    datagenerator = model.datagenerator

    if model.loaded == False:
        #todo: i dont think this condition ever happens
        log('do_inference recieved unloaded model')
        model = load_model()
    else:
        log('got loaded model for inference')

    log(type(model))

    ## just in case this hasnt already been done
    model.build(input_shape=(None, 64, 64, 3))

    #idk why
    if epoch is None:
        model.summary()

    #get a batch of images from the datagenerator
    images, labels = datagenerator.next()

    ## make an array of input images
    in_images =[]
    i = 0
    for item in images:

        pil_img = tf.keras.preprocessing.image.array_to_img(item)
        log(type(pil_img))

        dir = 'ae_samples/'

        if not os.path.exists(dir):
            os.makedirs(dir)

        in_images.append(pil_img)
        i = i + 1

    ## do the inference on the inputs
    result = model.predict(images, batch_size=batch_size)

    ## make an array of output images
    out_images = []
    i = 0
    for item in result:
        img = item
        img = img[:, :, 0:3] #/ 127.0

        pil_img = tf.keras.preprocessing.image.array_to_img(img)
        out_images.append(pil_img)
        i = i + 1

    #concatenate the two arays of images and add to list
    a = 1
    concatted = []
    for item in zip(in_images, out_images):
        #get_concat_h(item[0], item[1]).save(dir + '/ae_sample-' + str(a) + '.jpeg')
        new_img = get_concat_h(item[0], item[1])
        concatted.append(new_img)
        a = a + 1

    #concatenate images into rows 4 sets wide
    rows = []
    for i in range(0, len(concatted)-1, 4):
        rows.append(concat_four_h(concatted[i], concatted[i+1], concatted[i+2], concatted[i+3]))

    #concatenate the rows vertically t oform the final image
    first_img = rows[0]
    for i in range(1, len(rows)):
        first_img = get_concat_v(first_img, rows[i])

    #save the image to disk
    first_img.save('ae_samples/epoch_' + str(epoch) + '.jpeg')
    #return first_img