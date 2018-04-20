from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Conv2D, Reshape
from keras.layers import Lambda, Cropping2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import optimizers
import numpy as np
import h5py
import csv
import cv2
import os.path
import tensorflow as tf


import math
import matplotlib.pylab as plt
from keras.backend import tf as ktf

samples = []
with open('./SDC_Term1_Data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)


def resize_images(image):
    image = image[60:135,:,:]
    image = cv2.resize(image,(200, 66), interpolation = cv2.INTER_AREA)
    return image


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(1, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steers = []
            for batch_sample in batch_samples:
                name_center = './SDC_Term1_Data3/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name_center)
                # plt.imshow(center_image)
                # plt.show()
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
                # plt.imshow(center_image)
                # plt.show()
                center_angle = float(batch_sample[3])
                name_left = './SDC_Term1_Data3/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name_left)
                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
                name_right = './SDC_Term1_Data3/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name_right)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
                left_angle = center_angle + 0.25
                right_angle = center_angle - 0.25
                center_image = resize_images(center_image)
                left_image = resize_images(left_image)
                right_image = resize_images(right_image)
                images.append(center_image)
                steers.append(center_angle)
                # plt.imshow(center_image)
                # plt.show()
                flip_image = cv2.flip(center_image, 1)
                images.append(flip_image) # flip augement
                steers.append(-1*center_angle) # flip augement
                # plt.imshow(cv2.flip(center_image, 1))
                # plt.show()
                images.append(left_image)
                steers.append(left_angle)
                images.append(right_image)
                steers.append(right_angle)
            X_train = np.array(images)
            y_train = np.array(steers)

            yield shuffle(X_train, y_train)

source_path = './SDC_Term1_Data3/IMG/' + train_samples[0][0].split('/')[-1]
image_c = cv2.imread(source_path)
image_c = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)
image_c = resize_images(image_c)
rows, cols, channels = image_c.shape

batch_size = 64
# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

if os.path.isfile('./model.h5'):
    print("I restored from trained model")
    model = load_model('./model.h5')
else:
    print("I didn't restore")

    # new NVidia model
    input_shape = (rows, cols, channels)
    model = Sequential()
    # model.add(Cropping2D(cropping=((53, 22), (0, 0)), input_shape=input_shape))
    # model.add(Lambda(resize_images))
    # model.add(Lambda(lambda image: resize_images(image)))
    model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=input_shape))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
    dropout = 0.5

    for i in range(len(nb_filters)):
        model.add(Conv2D(nb_filters[i],
                        kernel_size[i],
                        padding=padding[i],
                        strides=strides[i],
                        activation='elu'))

    # flatten layer
    model.add(Flatten())
    model.add(Dropout(dropout))
    # fully connected layers with dropout
    neurons = [100, 50, 10, 1]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l]))

    # optimizer = optimizers.Adam(lr=0.0001)
    model.compile(optimizer='adam',
                  loss='mse')

print('Now I am training the model, my total sample count is:', format(4*len(train_samples)))

model_history = model.fit_generator(train_generator,
                    steps_per_epoch=len(train_samples)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_samples)/batch_size,
                    epochs=5,
                    verbose=1)

model.save('./model.h5')
print('Model saved')

import matplotlib.pyplot as plt
# %matplotlib inline
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
