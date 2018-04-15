import csv
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D, MaxPooling2D, Dropout, Convolution2D
import h5py
from sklearn.utils import shuffle

samples = []
with open('C:\\SDC_Term1_Data\\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import numpy as np
import sklearn
from sklearn.utils import shuffle

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = 'C:\\SDC_Term1_Data\\IMG\\' + batch_sample[i].split('/')[-1]
                    image = cv2.imread(source_path)
                    images.append(image)
                    angle = (i == 1) * 0.2 + float(line[3]) + (i == 2) * (-0.2)
                    angles.append(angle)
                # name = batch_sample[0].split('/')[-1]
                # center_image = cv2.imread(name)
                # center_angle = float(batch_sample[3])
                # name_left = batch_sample[1].split('/')[-1]
                # left_image = cv2.imread(name_left)
                # name_right = batch_sample[2].split('/')[-1]
                # right_image = cv2.imread(name_right)
                # center_angle = float(batch_sample[3])
                # left_angle = float(batch_sample[3])
                # right_angle = float(batch_sample[3])
                # images.append(center_image)
                # angles.append(center_angle)
                # images.append(left_image)
                # angles.append(left_angle)
                # images.append(right_image)
                # angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

from keras.layers import Lambda, Cropping2D
ch, row, col = 3, 80, 320  # Trimmed image format
# new NVidia model
model = Sequential()
model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# convolution layers with dropout
nb_filters = [24, 36, 48, 64, 64]
kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
same, valid = ('same', 'valid')
padding = [valid, valid, valid, valid, valid]
strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
dropout = 0.4

for l in range(len(nb_filters)):
    model.add(Convolution2D(nb_filters[l],
                            kernel_size[l][0], kernel_size[l][1],
                            border_mode=padding[l],
                            subsample=strides[l],
                            activation='elu'))
    model.add(Dropout(dropout))

# flatten layer
model.add(Flatten())

# fully connected layers with dropout
neurons = [100, 50, 10]
for l in range(len(neurons)):
    model.add(Dense(neurons[l], activation='elu'))
    model.add(Dropout(dropout))

# logit output - steering angle
model.add(Dense(1, activation='elu', name='Out'))
from keras import optimizers
optimizer = optimizers.Adam(lr=0.001)
model.compile(optimizer=optimizer,
              loss='mse')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


print('Now I am training the model')

model_history = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=2,
                    verbose=1)

model.save('C:\\Users\\zypch\\Documents\\Learning\\SDC_Term1\\CarND-Behavioral-Cloning-P3\\test_generator.h5')
print('Model saved')