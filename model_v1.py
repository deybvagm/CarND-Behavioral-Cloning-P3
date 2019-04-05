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

data_path = './data/IMG/'

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

lines = []
with open('./data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

index = 10
print('Total number of lines: {}'.format(len(lines)))
print('Line example: {}'.format(lines[index]))

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def generator(samples, batch_size=32):
    """
    This function is the generator that by default process 32 images  at each step of the training algorithm
    Each RGB image is resized to 66x200 px and then converted to a YUV color space because the CNN works that way
    """
    num_samples = len(samples)
    while 1:
        samples = shuffle(samples)
        for offset in range (0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                filename = batch_sample[0].split('/')[-1]
                center_image = cv2.imread(data_path + filename) #TODO this is in format BGR
                resize = (200, 66)
                center_image = cv2.resize(center_image, resize, interpolation=cv2.INTER_AREA)
                center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2YUV)                                
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            X_data = np.array(images)
            y_data = np.array(angles)
            yield shuffle(X_data, y_data)

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
history_obj = model.fit_generator(train_generator, steps_per_epoch=len(train_samples), validation_data=validation_generator, validation_steps=len(validation_samples), epochs=3, verbose=1)

model.save('model/model_v1.h5')
