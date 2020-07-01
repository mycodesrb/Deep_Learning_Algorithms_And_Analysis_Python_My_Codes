#!/usr/bin/env python
# coding: utf-8

# Here I have mplemented CNN on the CIFAR-10 dataset. It is a collection of images that are commonly used to train machine
# learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research.
# The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent
# airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. 50,000 images for training and 10,000 for test

# import required package
import pandas as pd
import numpy as np
from keras.datasets import cifar10
import matplotlib.pyplot as plt

# Load the dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

# Output classes
np.unique(y_test)

# We will follow the general practice of reshape the inputs and hot encoding the outputs
print("Before Reshaping: ", np.min(X_train), np.max(X_train))
X_train1 = X_train.reshape(50000, 32,32, 3).astype("float32")
X_train1 = X_train1/255
print("After Reshaping: ", np.min(X_train1), np.max(X_train1))

X_test1 = X_test.reshape(10000, 32,32,3).astype('float32')
X_test1 = X_test1/255
print("After Reshaping: ", np.min(X_test1), np.max(X_test1))

# Hot Encode Output
from keras.utils import np_utils

y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)
print(y_train.shape, y_test.shape)

# Import required libs for building the CNN model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten

# Build the CNN model
model =  Sequential()
# 1st CNN layer
model.add(Conv2D(32, (3,3), input_shape=(32,32,3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(3,3)))
# 2nd CNN layer
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(1000, activation='relu')) # for 1000 iterations
model.add(Dense(10, activation='softmax')) # for 10 output classes

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Fit data
hist = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=2, verbose=1)

# check accuracy and loss
print(hist.history['accuracy'])
print(hist.history['loss'])

# predict
res = np.argmax(model.predict(X_test1[17:18]))
res

# Check
plt.imshow(X_test[17])

# verify
y_test[17]

Bingo..!
