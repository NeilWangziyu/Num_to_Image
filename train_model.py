import keras
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.layers import Input
from keras.models import Model, Sequential
from keras.layers.core import Reshape, Dense, Dropout, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.datasets import mnist
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers
from keras.utils.np_utils import to_categorical
from keras.models import load_model
import argparse

dLosses = []
gLosses = []

def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('{}/dcgan_loss_epoch_%d.png'.format(image_folder) % epoch)


def plotGeneratedImages(epoch, examples=100, dim=(10, 10), figsize=(10, 10)):
    pickup_batch = np.arange(0, examples)

    our_input = to_categorical(y_test[pickup_batch], inputDim)

    generatedImages = generator.predict(our_input)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('{}/dcgan_generated_image_epoch_%d.png'.format(image_folder) % epoch)


def saveModels(epoch):
    generator.save('{}/dcgan_generator_epoch_%d.h5'.format(model_folder) % epoch)
    discriminator.save('{}/dcgan_discriminator_epoch_%d.h5'.format(model_folder) % epoch)


def train(epochs=1, batchSize=128):
    batchCount = X_train.shape[0] // batchSize
    print('Epochs:', epochs)
    print('Batch size:', batchSize)
    print('Batches per epoch:', batchCount)

    for e in range(1, epochs + 1):
        print('-' * 15, 'Epoch %d' % e, '-' * 15)

        for _ in range(batchCount):
            # Get a random set of input noise and images

            pickup_batch = np.random.randint(0, X_train.shape[0], size=batchSize)

            our_input = to_categorical(y_train[pickup_batch], inputDim)

            imageBatch = X_train[pickup_batch]

            generatedImages = generator.predict(our_input)
            X = np.concatenate([imageBatch, generatedImages])

            image_outputs = to_categorical(y_train[pickup_batch], inputDim + 1)
            gen_image_outputs = to_categorical([inputDim] * batchSize, inputDim + 1)
            yDis = np.vstack((image_outputs, gen_image_outputs))

            discriminator.trainable = True
            dloss = discriminator.train_on_batch(X, yDis)

            our_input = to_categorical(y_train[pickup_batch], inputDim)

            yGen = to_categorical(y_train[pickup_batch], inputDim + 1)

            discriminator.trainable = False
            gloss = gan.train_on_batch(our_input, yGen)

        dLosses.append(dloss)
        gLosses.append(gloss)

        if e == 1 or e % 5 == 0:
            plotGeneratedImages(e)
            saveModels(e)
    plotLoss(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train the GAN model')
    parser.add_argument('--epoch', default="25", help="the epoch number of training")
    args = parser.parse_args()
    epoch_number = int(args.epoch)

    K.set_image_dim_ordering('th')

    inputDim = 10

    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = X_train[:, np.newaxis, :, :]

    adam = Adam(lr=0.0002, beta_1=0.5)

    generator = Sequential()
    generator.add(Dense(128 * 7 * 7, input_dim=inputDim, kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    generator.add(LeakyReLU(0.2))
    generator.add(Reshape((128, 7, 7)))
    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    generator.add(LeakyReLU(0.2))
    generator.add(UpSampling2D(size=(2, 2)))
    generator.add(Conv2D(1, kernel_size=(5, 5), padding='same', activation='tanh'))
    generator.compile(loss='mse', optimizer=adam)

    discriminator = Sequential()
    discriminator.add(Conv2D(64, kernel_size=(5, 5), strides=(2, 2), padding='same', input_shape=(1, 28, 28),
                             kernel_initializer=initializers.RandomNormal(stddev=0.02)))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same'))
    discriminator.add(LeakyReLU(0.2))
    discriminator.add(Dropout(0.3))
    discriminator.add(Flatten())
    discriminator.add(Dense(inputDim + 1, activation='softmax'))
    discriminator.compile(loss='categorical_crossentropy', optimizer=adam)

    discriminator.trainable = False

    ganInput = Input(shape=(inputDim,))
    x = generator(ganInput)
    ganOutput = discriminator(x)
    gan = Model(inputs=ganInput, outputs=ganOutput)
    gan.compile(loss='categorical_crossentropy', optimizer=adam)

    image_folder = 'images_saved_for_each_epoch'

    if not os.path.exists(image_folder):
        os.mkdir(image_folder)

    model_folder = 'model_saved_for_each_epoch'

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)


    train(epoch_number, 128)

    generatedImages = generator.predict(to_categorical(np.array([8, 5, 4, 8, 1]), inputDim))

    dim = (10, 10)
    figsize = (10, 10)

    plt.figure(figsize=figsize)
    for i in range(generatedImages.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generatedImages[i, 0], interpolation='nearest', cmap='gray_r')
        plt.axis('off')
    plt.title("Sample image")
    plt.tight_layout()
    plt.show()
