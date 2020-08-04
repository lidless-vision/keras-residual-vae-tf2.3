"""

This script was inspired mostly from code in the keras and tensorflow documentation.
the original header is still preserved below, i think ive documented all the code i copied in the comments
Last modified: August 3rd, 2020


Title: Multi-GPU and distributed training
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/28
Last modified: 2020/04/29
Description: Guide to multi-GPU & distributed training for Keras models.
"""

# some resnet stuff from https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba
# NOTE: this version of keras or whatever requires cuda==10.1


import os

import tensorflow as tf
from tensorflow import keras

from tensorflow import Tensor
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Flatten, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utilities import do_inference

import gc


TRAINING = True
verbose = False

#note: with this run  we did LEARNING_RATE = 0.00001 and steps_per_epoch = 1000 for 1200 "epochs"
#   then we added another 0 to the learning rate, which i think is what we're supposed to do.

LEARNING_RATE = 0.000001
steps_per_epoch = 1000

latent_dim = 8
batch_size = 64

checkpoint_dir = "./ae_checkpoints/"

target_size = (64, 64)  # image size in pixels
dropout_rate = 0.2

data_path = '/media/cameron/angelas files/celeb-ms-cropped-aligned/'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1";

# if you are using only 1 GPU, comment out this whole block:
multi_gpu_training = False
if multi_gpu_training:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    # tf.config.experimental.set_memory_growth(gpus[2], True)
    # tf.config.experimental.set_memory_growth(gpus[3], True)


# this is to fix our super cool keras memory leak
global epochs_since_restart
global how_many_epochs_before_restart

epochs_since_restart = 0
how_many_epochs_before_restart = 30


def restart():
    """
    # written by: plieningerweb
    # https://gist.github.com/plieningerweb/39e47584337a516f56da105365a2e4c6

    we use this function to restart this entire script after a specified number of "epochs"
    because there is a memoryleak somwhere in tf2/keras.
    tensorflow==2.2.0 seems to leak less when using keras data_generators
    """
    import sys
    print("argv was", sys.argv)
    print("sys.executable was", sys.executable)
    print("restart now")

    import os
    os.execv(sys.executable, ['python'] + sys.argv)


## Create the relu + BatchNormalization layer
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn


## Create a sampling layer
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


## Create a residual_block for the encoder and also for the descriminator
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    """
    this is the downsampling residual block, used by the encoder and also used by the descriminator

    note: i tried making everything here use Relu and it made black images after a few epochs
    note 2: using the "same" padding is apparently a problem (https://stackoverflow.com/questions/54643064/how-to-improve-the-accuracy-of-autoencoder)
    note:3 this network eventually got the exploding gradient problem and it was solved by lowering the learning rate
    """
    # log('residual_block')

    y = layers.Dropout(dropout_rate)(x)
    y = Conv2D(kernel_size=kernel_size,
               strides=(1 if not downsample else 2),
               filters=filters,
               padding='same')(y)
    y = relu_bn(y)
    y = layers.Dropout(dropout_rate)(y)
    y = Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding='same')(y)

    if downsample:
        x = layers.Dropout(dropout_rate)(x)
        x = Conv2D(kernel_size=2,
                   strides=2,
                   filters=filters)(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


# Create a transposed residual block for the generator/decoder
def residual_block_transpose(x: Tensor, upsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    """
    this is the upsampling (transposed) residual block, used by the part known as the decoder/generator
    """

    y = layers.Dropout(dropout_rate)(x)
    y = Conv2DTranspose(kernel_size=kernel_size,
                        strides=(1 if not upsample else 2),
                        filters=filters,
                        padding='same')(y)

    y = relu_bn(y)
    y = layers.Dropout(dropout_rate)(y)
    y = Conv2DTranspose(kernel_size=kernel_size,
                        strides=1,
                        filters=filters,
                        padding='same')(y)

    if upsample:
        x = layers.Dropout(dropout_rate)(x)
        x = Conv2DTranspose(kernel_size=2,
                            strides=2,
                            filters=filters)(x)

    out = Add()([x, y])
    out = relu_bn(out)
    return out


def get_encoder(latent_dim, num_filters, training=True):
    """sets up the encoder model"""
    encoder_inputs = Input(shape=(64, 64, 3))

    if training:
        # if training, insert this dropout layer so the model can learn to denoise or something
        x = layers.Dropout(dropout_rate)(encoder_inputs)
        x = BatchNormalization()(x)
    else:
        x = BatchNormalization()(encoder_inputs)

    x = Conv2D(kernel_size=2,
               strides=2,
               filters=num_filters)(x)
    x = relu_bn(x)

    num_blocks_list = [10, 10, 10]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            x = residual_block(x, downsample=(j == 0 and i != 0), filters=num_filters)
        num_filters /= 2  # its important that the encoders params get smaller because its
        # an autoencoder, so we divide here instead of multiplying like in
        # the discriminator
    x = layers.Flatten()(x)
    x = layers.Dense(32, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
    encoder.summary()

    return encoder


def get_decoder(latent_dim, num_filters):
    """sets up the decoder model"""
    latent_inputs = keras.Input(shape=(latent_dim,))

    x = layers.Dense(8 * 8 * num_filters, activation="relu")(latent_inputs)
    x = layers.Reshape((8, 8, num_filters))(x)

    num_blocks_list = [10, 10, 10]
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        for j in range(num_blocks):
            x = residual_block_transpose(x, upsample=(j == 0 and i != 0), filters=num_filters)
        num_filters *= 2

    x = residual_block_transpose(x, upsample=(True), filters=num_filters)
    # x = residual_block_transpose(x, upsample=(True), filters=num_filters)

    decoder_outputs = layers.Conv2DTranspose(3, 3, activation="sigmoid", padding="same")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
    decoder.summary()

    return decoder


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.datagenerator = None
        self.loaded = False

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                keras.losses.binary_crossentropy(data, reconstruction)
            )
            reconstruction_loss *= 64 * 64
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        del data, grads
        gc.collect()  # supposedly this will fix the memoryleak (https://github.com/tensorflow/tensorflow/issues/35030)

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs, training=True):
        # todo: this doesnt seem right, but yet it still works.

        latents = self.encoder(inputs)
        result = self.decoder(latents[0])

        gc.collect()  # supposedly this will fix the memoryleak (https://github.com/tensorflow/tensorflow/issues/35030)
        return result


class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print("End epoch {} generating samples... \n".format(epoch + 1))
        do_inference(self.model, epoch=epoch, batch_size=batch_size)

        global epochs_since_restart
        epochs_since_restart += 1

        print(
                'epochs since last restart: ' + str(epochs_since_restart) +
                ' epochs remaining until next restart: ' + str(how_many_epochs_before_restart)
              )

        if epochs_since_restart == how_many_epochs_before_restart:
            restart()
            exit()

        gc.collect()  # supposedly this will fix the memoryleak (https://github.com/tensorflow/tensorflow/issues/35030)


def load_model(training=True):
    """ builds the model and then loads the most recent weights from disk
    if they exist """

    print('compliling model...')
    model = compile_model(latent_dim=latent_dim, training=training)

    print('trying to load pretrained weights from disk...')
    checkpoints = [checkpoint_dir + name for name in os.listdir(checkpoint_dir)]

    if checkpoints != []:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print("restoring from checkpoint: ", latest_checkpoint)

        # parse the filename to get the epoch number of the last checkpoint
        current_epoch = str(latest_checkpoint).split('cpkt-')[1]
        current_epoch = int(current_epoch.split('.h5')[0])

        # load the weights from the checkpoint file
        model.load_weights(latest_checkpoint)
        print('loaded model.')
    else:
        current_epoch = 0
        print('no checkpoint found, using untrained model.')

    model.loaded = True

    return model, current_epoch


def compile_model(latent_dim, training=True):
    ## Build the encoder
    encoder = get_encoder(latent_dim=latent_dim, training=training, num_filters=128)

    ## Build the decoder
    decoder = get_decoder(latent_dim=latent_dim, num_filters=32)

    ## Combine both into one VAE model
    model = VAE(encoder, decoder)
    optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
    model.compile(optimizer=optimizer)

    model.build(input_shape=(None, 64, 64, 3))
    model.loaded = True

    return model


def run_training(model, current_epoch, epochs=10000):
    # log(type(model))
    print('steps per epoch = ' + str(steps_per_epoch))

    callbacks = [
        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        # keras.callbacks.experimental.BackupAndRestore(backup_dir='backup/'),
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/cpkt-{epoch}.h5", save_freq="epoch"
        ),
        keras.callbacks.TensorBoard(
            log_dir='./logs', update_freq=100, profile_batch='1,100000'
        ),
        CustomCallback()
    ]

    print('starting training from epoch ' + str(current_epoch))

    model.fit(
        datagenerator, verbose=1,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        initial_epoch=current_epoch, batch_size=batch_size
    )


def get_datagenerator(path):
    # create a data generator

    # note: the folder has to have folders of images, these are the class labels
    #       but this model ignores class labels
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    train_generator = datagen.flow_from_directory(path, batch_size=batch_size, shuffle=True,
                                                  class_mode='input', color_mode="rgb"
                                                  , target_size=(64, 64))
    return train_generator


if __name__ == '__main__':

    # Prepare a directory to store all the checkpoints.
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    """
    # Everything that creates variables should be under the strategy scope.
    # In general this is only model construction & `compile()`.
    """
    strategy = tf.distribute.MirroredStrategy()
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    with strategy.scope():

        # print('loading dataset ')
        # datagenerator = load_dataset(data_path)

        print('starting the datagenerator')

        datagenerator = get_datagenerator(data_path)

        model, current_epoch = load_model(training=TRAINING)

        if TRAINING == True:
            print('running training')
            model.datagenerator = datagenerator

            if current_epoch == 0:
                do_inference(model, None, batch_size=batch_size)
            else:
                do_inference(model, current_epoch, batch_size=batch_size)

            run_training(model, current_epoch, epochs=10000)
        else:
            print("just doing inference")
            model.datagenerator = datagenerator
            model = do_inference(model, epoch=None, batch_size=batch_size)
