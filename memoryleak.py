#todo: This branch is the version which was modified to use the new tf.data
# functions where the model.fit() function is passed a dataset rather than a data_generator
#       this is preferable so we dont have to wait 10 minutes for
#       the datagenerator to scan the directories every time
#       we are implementing an amazing workaround where we just restart this whole program
#       in order to avoid the memory leak
#       https://github.com/tensorflow/tensorflow/issues/35030

#todo: just kidding, this uses up the enirety of my ram before even one epoch finishes

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.layers import Input, Dense, Flatten, Reshape


import numpy
import os
import PIL.Image as Image


batch_size = 32

if __name__ == '__main__':


    if not os.path.exists('images/'):
        os.makedirs('images/')


    print('generating test dataset of random images')
    for i in range(0, 100):

        if not os.path.exists('images/' + str(i) + '/'):
            os.makedirs('images/' + str(i) + '/')

        for x in range(0, 100):
            imarray = numpy.random.rand(64, 64, 3) * 255
            im = Image.fromarray(imarray.astype('uint8')).convert('RGB')
            im.save('images/' + str(i) + '/' + str(x) + '.jpg')


    my_dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'images/', label_mode=None, class_names=None,
        color_mode='rgb', batch_size=32, image_size=(64, 64), shuffle=False, seed=None,
        validation_split=None, subset=None, interpolation='bilinear', follow_links=False
    )


    model = tf.keras.Sequential([
    Input(shape=(64, 64, 3)),
    Dense(64 * 64 * 3, activation='relu'),
    Dense(2, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='sigmoid'),
    Reshape(target_shape=[64, 64, 3])
    ])


    # model.compile(
    #     loss='sparse_categorical_crossentropy',
    #     optimizer=tf.keras.optimizers.Adam(0.001)
    # )
    #
    #
    #
    # model.fit(
    #     my_dataset, verbose=1,
    #     steps_per_epoch=1000,
    #     epochs=1000,
    #     batch_size=batch_size
    # )
    #

