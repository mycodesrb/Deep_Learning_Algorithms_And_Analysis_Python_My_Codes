#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Here, I have used the mnist dataset from Keras library. We will load this data, split in train and test and process the data
# in order to make it in a proper format to input to the CNN model. Then we will train the model and we will check the loss
# and accuracy of the CNN model. Post that we will try to predict whether the model predicts the data corectly.

# About keras: Keras is a high-level neural network API focused on user friendliness, fast prototyping, modularity and
# extensibility. 

# PACKAGE CONTENTS boston_housing, cifar, cifar10, cifar100, fashion_mnist, imdb, mnist, reuters out of which we used mnist
# mnist dataset: MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images 
# are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker. 


# In[11]:


#import required libs
import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[4]:


#load data
(X_train, y_train),(X_test, y_test) = mnist.load_data()


# In[5]:


print(X_train.shape, X_test.shape)


# In[6]:


print(y_train.shape, y_test.shape)


# In[19]:


plt.imshow(X_train[0])
print(y_train[0])


# In[21]:


plt.imshow(X_train[1987])
print(y_train[1987])


# In[23]:


# process the input data with proper reshaping and encoding from min to max as 0 to 1
print("Before Reshaping X_train: ", np.min(X_train), np.max(X_train))
X_train1 = X_train.reshape(60000, 28,28,1).astype('float32') # 1 for gray scale
X_train1 = X_train1/255
print("After Reshaping X_train: ", np.min(X_train1), np.max(X_train1))

X_test1 = X_test.reshape(10000, 28,28,1).astype("float32")
X_test1 = X_test1/255


# In[25]:


# hot encode output data
from keras.utils import np_utils

y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)


# In[22]:


# import packages for CNN model 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Flatten


# In[31]:


# build model
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(3,3)))

model.add(Flatten())

model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# In[34]:


# fit data
hist = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=2,  batch_size=1024, verbose=1)


# In[36]:


# check
print(hist.history['accuracy'])
print(hist.history['loss'])


# In[39]:


# predict
res = np.argmax(model.predict(X_train1[300:301]))
res


# In[40]:


# Check
plt.imshow(X_train[300])


# In[41]:


# verify
y_train[300]


# In[ ]:


Bingo..!!

