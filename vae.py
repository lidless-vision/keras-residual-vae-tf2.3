



# the purpose of this autoencoder is actually implement some research where they use a pretrained autoencoder as the
# generator in a GAN, and claim that it prevents mode collapse and reduces training time.
# i am going to upload that part as soon as i can figure out how to make it work.
# ive been having a hard time finding a gan architecture that works, maybe ill have to actually read the paper
# that paper is here https://arxiv.org/abs/2002.02112

#todo: this code was basically assembled from the keras documentation and maybe some blog posts, i need to create
#       a proper bibliography for this code. shown below is the original header from François Chollet

"""
Title: Multi-GPU and distributed training
Author: [fchollet](https://twitter.com/fchollet)
Date created: 2020/04/28
Last modified: 2020/04/29
Description: Guide to multi-GPU & distributed training for Keras models.
"""

# some resnet stuff from https://towardsdatascience.com/building-a-resnet-in-keras-e8f1322a49ba



# NOTE: this version of keras or whatever (tensorflow==2.2.0) requires cuda==10.1

# NOTE: this script seems to work fine on tensorflow 2.3.0, and also tf 2.4.0 which has some
#       cool features but also some bad memory leaks that made it unusable. #todo: post that version too


from utilities import *
import os

import tensorflow as tf
from tensorflow import keras

from tensorflow import Tensor
from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Add
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, AveragePooling2D, Flatten, Conv2DTranspose
from tensorflow.keras.preprocessing.image import ImageDataGenerator


TRAINING = True
verbose = False

LEARNING_RATE = 0.00001
steps_per_epoch = 200

latent_dim = 8
batch_size = 64

checkpoint_dir = "./ae_checkpoints/"

target_size = (64, 64)  # image size in pixels
dropout_rate = 0.2

#here goes the path to the ms celeb dataset
data_path = '/media/cameron/angelas files/celeb-ms-cropped-aligned/'


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"



###################################### if you have only 1 GPU just comment this out:
multi_gpu_training = False
if multi_gpu_training:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.config.experimental.set_memory_growth(gpus[1], True)
    # tf.config.experimental.set_memory_growth(gpus[2], True)
    # tf.config.experimental.set_memory_growth(gpus[3], True)


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
    #log('residual_block')

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

#Create a transposed residual block for the generator/decoder
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

def get_encoder(latent_dim,num_filters, training=True):
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
    #x = residual_block_transpose(x, upsample=(True), filters=num_filters)

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
        log('train step')

        log(type(data))
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            log(type(data))
            log(data.shape)
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
        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
        }

    def call(self, inputs, training=True):

        #todo: this doesnt seem right, yet it still works.

        latents = self.encoder(inputs)
        result = self.decoder(latents[0])

        return result

class CustomCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        #keys = list(logs.keys())
        print("End epoch {} generating samples... ".format(epoch + 1))
        do_inference(self.model, epoch=epoch)


def load_model(training=True):
    """ builds the model and then loads the most recent weights from disk
    if they exist """

    print('compliling model...')
    model = compile_model(latent_dim=latent_dim, training=training)

    print('trying to load pretrained weights from disk...')
    checkpoints = [checkpoint_dir + name for name in os.listdir(checkpoint_dir)]

    log(checkpoints)

    if checkpoints != []:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)

        print("restoring from checkpoint: ", latest_checkpoint)

        # parse the filename to get the epoch number of the last checkpoint
        current_epoch = str(latest_checkpoint).split('ckpt-')[1]
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
    model.loaded = True

    return model

def run_training(model, current_epoch, epochs=10000):

    log(type(model))
    print('steps per epoch = ' + str(steps_per_epoch))

    callbacks = [

        # this is a new feature in tf 2.3 keras and it does essentially the same as the one below but in one line of code
        # but it only saves the most recent checkpoints so... lets try it out.
        keras.callbacks.experimental.BackupAndRestore(backup_dir='backup/'),

        # This callback saves a SavedModel every epoch
        # We include the current epoch in the folder name.
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir + "/checkpoint-{epoch}.h5", save_freq="epoch"
        ),

        #this saves info to tensorboard so we can watch the graphs
        keras.callbacks.TensorBoard(
            log_dir='./logs', update_freq=100, profile_batch='1,100000'
        ),
        CustomCallback()
    ]

    print('starting training from epoch ' + str(current_epoch))
    log(type(model))

    model.fit_generator(
        datagenerator, verbose=1,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=callbacks,
        initial_epoch=current_epoch
    )

def get_datagenerator(path):
        # create a data generator

        # note: the folder has to have folders of images, these are the class labels
        #       but this model ignores class labels
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_generator = datagen.flow_from_directory(path, batch_size=batch_size, shuffle=False,
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

        print('starting the datagenerator')
        datagenerator = get_datagenerator(data_path)

        model, current_epoch = load_model(training=TRAINING)

        if TRAINING == True:
            print('running training')
            model.datagenerator = datagenerator

            if current_epoch == 0:
                do_inference(model, None)
            else:
                do_inference(model, current_epoch)

            run_training(model, current_epoch, epochs=10000)
        else:
            print("just doing inference")
            model.datagenerator = datagenerator
            model = do_inference(model, epoch=None)

