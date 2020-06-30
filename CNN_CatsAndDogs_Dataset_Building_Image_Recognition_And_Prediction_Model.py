#!/usr/bin/env python
# coding: utf-8

# Here I have built a Convolution Neural Network Model and trained it with images of Cats and Dogs and finally tried to 
# predict a given image.

# Dataset: Any image Dataset can be used where it gives images all with same resolution and names in a sequence. You can 
# download as per yor intrest. I have used Cats and Dogs images. It has 4000 gray scale images each for Cats and Dogs, placed 
# in respective 'cats' and 'dogs' folders and named in a sequence.

# Basic imports
import numpy as np # to operate on arrays
import os # to use paths of folders
import cv2 # to read and resize the images
import matplotlib.pyplot as plt

#Load Data from the main folders inside which cats and dogs folders are present
lsmain = os.listdir("D:/2020/Python/DL/DL datasets/CNN_Cat_Dog/training_set") #main dir
lsmain

print(lsmain.index('cats'))
print(lsmain.index('dogs'))

lsimages, lslabels=[],[]

for i in lsmain:
    lssubmain = os.listdir("D:/2020/Python/DL/DL datasets/CNN_Cat_Dog/training_set/"+i)
    for s in lssubmain:
        path = "D:/2020/Python/DL/DL datasets/CNN_Cat_Dog/training_set/"+i+"/"+s
        img = cv2.imread(path,0) #read images, with 0 as grayscale
        imgs = cv2.resize(img, (80,80)) # resize all images in one size. we are reducing the size for easy running
        lsimages.append(imgs)
        lslabels.append(lsmain.index(i))
        #plt.imshow(imgs)

print(len(lsimages), len(lslabels))

lslabels[3999] # outputs

plt.imshow(lsimages[3999]) #Example check

plt.imshow(lsimages[4000]) # example check

(np.array(lsimages)).shape # shape of input list

arr = np.array(lsimages) # convert input list into array
print("Before reshaping: ",arr.shape)
X = arr.reshape(arr.shape[0], 80, 80, 1).astype('float') #reshape it: For image data, the channel dimension is 1 for grayscale
                                                        # images
X1 = X/255 # convert the min and max in range 0 to 1 to limit the spread and for the ease of clculation
print("After reshaping: ", X1.shape, np.min(X1), np.max(X1))

# X1[11] #check
plt.imshow(lsimages[11]) #Check

from keras.utils import np_utils # to hot encode output

y = np_utils.to_categorical(np.array(lslabels))
y.shape

y[4000] #check

#split in train and test
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.20)

print(X_train.shape, y_train.shape)

print(X_test.shape, y_test.shape)

#Apply Convolution
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D # for convolution layer
from keras.layers.convolutional import MaxPooling2D # to pool the data
from keras.layers import Flatten

model = Sequential()

model.add(Conv2D(32, (3,3), input_shape = (80,80,1), activation='relu')) #added convolution layer1
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64, (3,3), activation='relu')) #added convolution layer2
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(128, (3,3), activation='relu')) #added convolution layer3
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # finally flatten the array 

model.add(Dense(1000, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, verbose=1)

hist.history['accuracy']

#read and process image to be predicted (from external source)
def get_img():
    i = "d:/c2.jpeg" # a cat image
    im = cv2.imread(i,0)
    img = cv2.resize(im, (80, 80))
    img_ar = np.array(img)
    im_x = img_ar.reshape(1, 80, 80, 1).astype('float')
    image = im_x/255
    #print(image)
    return image

res = model.predict(get_img()) # predict external image

np.argmax(res) #gives calss 0,1 depending on the prediction (index 0 is for cats, predicted correctly)

res = model.predict(X_train[6000:6001]) # predict 6000th image from the available data

np.argmax(res)

plt.imshow(lsimages[6000]) # its a dog, so predicted correctly
