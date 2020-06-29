#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Here, I have used the mnist dataset from Keras library. We will load this data, split in train and test and process the data
# in order to make it in a proper format to input to the ANN model. Then we will train the model and we will check the loss
# and accuracy of the ANN model. Post that we will try to predict whether the model predict the data.

# About in keras: is a high-level neural network API focused on user friendliness, fast prototyping, modularity and
# extensibility. 

# PACKAGE CONTENTS boston_housing, cifar, cifar10, cifar100, fashion_mnist, imdb, mnist, reuters out of which we used mnist
# mnist dataset: MNIST contains 70,000 images of handwritten digits: 60,000 for training and 10,000 for testing. The images 
# are grayscale, 28x28 pixels, and centered to reduce preprocessing and get started quicker. 


# In[4]:


# import required packages
from keras.datasets import mnist
import matplotlib.pyplot as plt


# In[5]:


# Load dataset in train and test
(X_train, y_train),(X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[6]:


# Check what kind of images are there in mnist
#plt.imshow(X_train[2000])
plt.imshow(X_test[0])


# In[7]:


# Now, first step is to convert train/test input images data from 3D form to 2D. We will process the data and convert it to 2D
print(X_train.shape, X_test.shape)
num_of_pixels = X_train.shape[1]*X_train.shape[2]
print(num_of_pixels)


# In[8]:


import numpy as np

X_train1 = X_train.reshape(X_train.shape[0], num_of_pixels).astype('float')
X_test1 = X_test.reshape(X_test.shape[0], num_of_pixels).astype('float')
print("2D Shape post reshaping: ",X_train1.shape, X_test1.shape)


# In[9]:


# Why we made array as float ? Float is made as we need to scale input data (min to max) as 0 to 1 
print(np.min(X_train1), np.max(X_train1)) # before conversion
X_train1 = X_train1/255
X_test1 = X_test1/255
print(np.min(X_train1), np.max(X_train1)) # post conversion: This makes data less spread as compared to 0-255 and calculations 
                                          # becomes easy


# In[10]:


# Now hot encode the output data, y_train and y_test: hot encoding is representing categorical values in binary vectors
from keras.utils import np_utils
# this encode values 0-9: In an array of 10 places, set 1 for value and 0 in rest of the places 


# In[11]:


print("Before Encoding y_train[0] is ", y_train[0])
y_train1 = np_utils.to_categorical(y_train)
y_test1 = np_utils.to_categorical(y_test)
print("Post Encoding y_train[0] is :", y_train1[0]) # just to show the example


# In[12]:


# Create Model and layers
from keras.models import Sequential 
from keras.layers import Dense 
# Sequental model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output
# tensor
# Dense layer is the regular deeply connected neural network layer


# In[13]:


#output Classes
num_classes = y_test1.shape[1]
num_classes


# In[23]:


# Create model
model = Sequential() # model created
model.add(Dense(1000, input_dim=num_of_pixels, activation='relu')) # layer for input added
model.add(Dense(num_classes, activation='softmax')) #layer for output added
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[24]:


# get summary
model.summary()


# In[29]:


# Fit data
hist = model.fit(X_train1, y_train1, validation_data=(X_test1, y_test1), epochs=2, verbose=1)


# In[31]:


# Check parameters
print(hist.history['accuracy'])
print(hist.history['loss'])
# See the loss is reduced and accuracy is increased from initial to final iteration


# In[41]:


# Predict
arr_result = model.predict(X_test1[4000:4001])#for 1st row, write as [0:1] instead of [0]
                            #for 2nd row, write as [1:2] instead of [1]...so on
arr_result    


# In[42]:


# Check the output
np.argmax(arr_result)


# In[43]:


# Check the image form
plt.imshow(X_test[4000])


# In[44]:


# Verify 
y_test[4000] #..Bingo..!!!


# In[ ]:




