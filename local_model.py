from scipy import ndimage
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Activation
from keras.layers import Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
import os

data_path = '/home/deyber/PycharmProjects/self-driving-car-nanodegree/simulator-linux/joystick-data'

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open(os.path.join(data_path, 'driving_log.csv')) as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

index = 10
print('Total number of lines: {}'.format(len(lines)))
print('Line example: {}'.format(lines[index]))

train_samples, validation_samples = train_test_split(lines, test_size=0.2)


def process_image(filepath):
    img = cv2.imread(filepath)  # TODO the image is in BGR format
    resize = (200, 66)
    img = cv2.resize(img, resize, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    return img


def flip_horizontally(images, angles):
    return [np.fliplr(img) for img in images], [-angle for angle in angles]


def generator(samples, batch_size=32):
    # In this case we'll have into account center, left and right camera, and flip images.
    # That is why in the next line batch_size is divided by 5 as each line of the file yield 6 images
    batch_size = batch_size // 5
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_img_filename = batch_sample[0].split('/')[-1]
                left_img_filename = batch_sample[1].split('/')[-1]
                right_img_filename = batch_sample[2].split('/')[-1]
                center_image = process_image(filepath=os.path.join(data_path, 'IMG', center_img_filename))
                left_image = process_image(filepath=os.path.join(data_path, 'IMG', left_img_filename))
                right_image = process_image(filepath=os.path.join(data_path, 'IMG', right_img_filename))
                center_angle = float(batch_sample[3])
                correction = 0.2
                left_angle = center_angle + correction
                right_angle = center_angle - correction

                base_images = [center_image, left_image, right_image]
                base_angles = [center_angle, left_angle, right_angle]

                new_imgs, new_angles = flip_horizontally(base_images, base_angles)

                appended_images = base_images + new_imgs
                appended_angles = base_angles + new_angles

                images = images + appended_images
                angles = angles + appended_angles

            X_data = np.array(images)
            y_data = np.array(angles)
            X_data, y_data = shuffle(X_data, y_data)
            #if len(y_data) <= 32:
            #    idxs = np.random.choice(len(y_data), 32, replace=False)
            #    X_data = X_data[idxs]
            #    y_data = y_data[idxs]
            yield X_data[0:32], y_data[0:32]


train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

input_shape = (66, 200, 3)
# CNN architecture

model = Sequential()
# model.add(Cropping2D(cropping=((25, 8), (0, 0)), input_shape=input_shape))
model.add(Lambda(lambda x: x / 127.5 - 1., input_shape=input_shape))
model.add(Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(filters=36, kernel_size=(5, 5), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(filters=48, kernel_size=(5, 5), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Convolution2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
history_obj = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
                                  validation_data=validation_generator, validation_steps=len(validation_samples),
                                  epochs=5, verbose=1)

model.save('model/local_model.h5')
