from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Convolution2D
from keras.layers import Lambda, Cropping2D, MaxPooling2D, ELU
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import load_model
from keras import optimizers
import numpy as np
import tensorflow as tf
import h5py
import csv
import cv2
import os.path
import math

samples = []
with open('C:\\SDC_Term1_Data1\\driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples[1:], test_size=0.2)

# image translation, referred from https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def trans_image(image, steer, trans_range):
    # Translation
    tr_x = trans_range*(np.random.uniform()-1/2)
    steer_ang = steer + tr_x/trans_range*2*.2
    tr_y = 10*np.random.uniform()-10/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows))
    return image_tr, steer_ang, tr_x

# randomly change the brightness of the image
def augment_brightness(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random_bright = .25+np.random.uniform()
    #print(random_bright)
    image1[:,:,2] = image1[:,:,2]*random_bright
    image1 = cv2.cvtColor(image1,cv2.COLOR_HSV2RGB)
    return image1

def process_newimage_file(name):
    # Preprocessing image
    image = cv2.imread(name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def random_distort(img, angle):
    '''
    method for adding random distortion to dataset images, including random brightness adjust, and a random
    vertical shift of the horizon position
    '''
    new_img = img.astype(float)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(-28, 28)
    if value > 0:
        mask = (new_img[:,:,0] + value) > 255
    if value <= 0:
        mask = (new_img[:,:,0] + value) < 0
    new_img[:,:,0] += np.where(mask, 0, value)
    # random shadow - full height, random left/right side, random darkening
    h,w = new_img.shape[0:2]
    mid = np.random.randint(0,w)
    factor = np.random.uniform(0.6,0.8)
    if np.random.rand() > .5:
        new_img[:,0:mid,0] *= factor
    else:
        new_img[:,mid:w,0] *= factor
    # randomly shift horizon
    h,w,_ = new_img.shape
    horizon = 2*h/5
    v_shift = np.random.randint(-h/8,h/8)
    pts1 = np.float32([[0,horizon],[w,horizon],[0,h],[w,h]])
    pts2 = np.float32([[0,horizon+v_shift],[w,horizon+v_shift],[0,h],[w,h]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    new_img = cv2.warpPerspective(new_img,M,(w,h), borderMode=cv2.BORDER_REPLICATE)
    return (new_img.astype(np.uint8), angle)

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(1, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            steers = []
            for batch_sample in batch_samples:
                # if float(batch_sample[3]) > 0.25:      # if the throttle is more than 3
                # for i in range(3):
                #     source_path = 'C:\\SDC_Term1_Data1\\IMG\\' + batch_sample[i].split('/')[-1]
                #     image = cv2.imread(source_path)
                #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                #     steer = (i == 1) * 0.25 + float(line[3]) + (i == 2) * (-0.25)
                #     image, steer, tr_x = trans_image(image, steer, 150)
                #     image = augment_brightness(image)
                #     # image = process_newimage_file(image)
                #     image, steer = random_distort(image, steer)
                #     images.append(image)
                #     steers.append(steer)
                name_center = 'C:\\SDC_Term1_Data1\\IMG\\' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name_center)
                center_angle = float(batch_sample[3])
                name_left = 'C:\\SDC_Term1_Data1\\IMG\\' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name_left)
                name_right = 'C:\\SDC_Term1_Data1\\IMG\\' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name_right)
                left_angle = float(batch_sample[3]) + 0.25
                right_angle = float(batch_sample[3]) - 0.25
                images.append(center_image)
                steers.append(center_angle)
                images.append(left_image)
                steers.append(left_angle)
                images.append(right_image)
                steers.append(right_angle)
            augmented_images, augmented_measurements = [], []
            for image, steer in zip(images, steers):
                augmented_images.append(image)
                augmented_measurements.append(steer)
                if abs(steer) > 0.15:  # only apply on the images with large steering
                    augmented_images.append(cv2.flip(image, 1))
                    augmented_measurements.append(steer*(-1))
            X_train = np.array(augmented_images)
            y_train = np.array(augmented_measurements)
            yield shuffle(X_train, y_train)

source_path = 'C:\\SDC_Term1_Data1\\IMG\\' + train_samples[0][0].split('/')[-1]
image_c = process_newimage_file(source_path)
rows, cols, channels = image_c.shape

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=64)
validation_generator = generator(validation_samples, batch_size=64)

if os.path.isfile('.\\nvidia_3.h5'):
    print("I restored from trained model")
    model = load_model('.\\nvidia_3.h5')
else:
    print("I didn't restore")

    # new NVidia model
    input_shape = (rows, cols, channels)
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=input_shape))
    model.add(Cropping2D(cropping=((60, 25), (0, 0))))
    model.add(Lambda(lambda x: tf.image.resize_images(x, (66,200))))

    # convolution layers with dropout
    nb_filters = [24, 36, 48, 64, 64]
    kernel_size = [(5, 5), (5, 5), (5, 5), (3, 3), (3, 3)]
    same, valid = ('same', 'valid')
    padding = [valid, valid, valid, valid, valid]
    strides = [(2, 2), (2, 2), (2, 2), (1, 1), (1, 1)]
    dropout = 0.7

    for l in range(len(nb_filters)):
        model.add(Convolution2D(nb_filters[l],
                                kernel_size[l][0],
                                kernel_size[l][1],
                                border_mode=padding[l],
                                subsample=strides[l],
                                activation='elu'))
        model.add(Dropout(dropout))

    # flatten layer
    model.add(Flatten())
    model.add(Dropout(dropout))
    # fully connected layers with dropout
    neurons = [100, 50, 10, 1]
    for l in range(len(neurons)):
        model.add(Dense(neurons[l]))

#     filter_size = 3
#     pool_size = (2, 2)
#     model.add(Lambda(lambda x: x/255.-0.5,input_shape=input_shape))
#     model.add(Convolution2D(3,1,1,
#                         border_mode='valid',
#                         name='conv0', init='he_normal'))
#     model.add(Convolution2D(32,filter_size,filter_size,
#                         border_mode='valid',
#                         name='conv1', init='he_normal'))
#     model.add(ELU())
#     model.add(Convolution2D(32,filter_size,filter_size,
#                         border_mode='valid',
#                         name='conv2', init='he_normal'))
#     model.add(ELU())
#     model.add(MaxPooling2D(pool_size=pool_size))
#     model.add(Dropout(0.5))
#     model.add(Convolution2D(64,filter_size,filter_size,
#                         border_mode='valid',
#                         name='conv3', init='he_normal'))
#     model.add(ELU())

#     model.add(Convolution2D(64,filter_size,filter_size,
#                         border_mode='valid',
#                         name='conv4', init='he_normal'))
#     model.add(ELU())
#     model.add(MaxPooling2D(pool_size=pool_size))
#     model.add(Dropout(0.5))
#     model.add(Convolution2D(128,filter_size,filter_size,
#                         border_mode='valid',
#                         name='conv5', init='he_normal'))
#     model.add(ELU())
#     model.add(Convolution2D(128,filter_size,filter_size,
#                         border_mode='valid',
#                         name='conv6', init='he_normal'))
#     model.add(ELU())
#     model.add(MaxPooling2D(pool_size=pool_size))
#     model.add(Dropout(0.5))
#     model.add(Flatten())
#     model.add(Dense(512,name='hidden1', init='he_normal'))
#     model.add(ELU())
#     model.add(Dropout(0.5))
#     model.add(Dense(64,name='hidden2', init='he_normal'))
#     model.add(ELU())
#     model.add(Dropout(0.5))
#     model.add(Dense(16,name='hidden3',init='he_normal'))
#     model.add(ELU())
#     model.add(Dropout(0.5))
#     model.add(Dense(1, name='output', init='he_normal'))

    optimizer = optimizers.Adam(lr=0.0001)
    model.compile(optimizer=optimizer,
                  loss='mse')

print('Now I am training the model')

model_history = model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples),
                    nb_epoch=5,
                    verbose=1)
model.save('.\\nvidia_3.h5')
print('Model saved')
