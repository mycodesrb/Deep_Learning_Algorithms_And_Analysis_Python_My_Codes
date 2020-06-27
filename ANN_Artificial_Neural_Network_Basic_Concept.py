#!/usr/bin/env python
# coding: utf-8

# Here I have shown the very basic construct of a Artificial Neural Network (ANN). ANN has inputs values/input layers which
# interacts with hidden layers (can be multiple) through some mathematical operation, gets adjusted according to some activation
# function and the procedure is repeated till the errors in the receieved output are are minimized w.r.t the expected output 
# We will see this basic flow implementation.
# Generally we have 3 Activation Functions: Sigmoid, Relu and Tanh. Here I have taken sigmoid function. Also, I have defined 
# a derivative function which will be used while calculating the adjustments.  

# import the required: We will manupulate data in arrays
import numpy as np

# define sigmoid and derivative functions
def sigmoid(y):
    return 1/(1+np.exp(-y))

def sigmoid_derivative(x):
    return x*(1-x)

# Take a random input array of size say (4,3)
training_ip = np.array([[0,0,1],
              [1,1,1],
              [1,0,1],
              [0,1,1]])

# Take an output array (we expect the achieve this output)
training_op = np.array([[0,1,1,0]]).T # Transpose: to match these values row by row with input values

# We need some weights in order to have a random number to me multiplied with the input layer to get an output
# We generate some random numbers, with some mathematical changes. These mathematical changes can be anything

synaptic_weights = 2*np.random.random((3,1))-1
synaptic_weights

# Build the algorithmic model
lsoutputs, lserrors=[],[] #to store outputs and errors, just to compare number of outputs and number of errors: should be equal
for iteration in range(50000):
    input_layer = training_ip #just for understanding
    output = sigmoid(np.dot(input_layer,synaptic_weights)) #get normalized output
    #print(output[0,:]) just checking first row
    lsoutputs.append(output[0,:]) # just appending 1st row instead of entire array
    errors = training_op-output
    lserrors.append(errors[0,:])# just appending 1st row instead of entire array
    # to change the received output so as to make closed to the expected output, we need to adjust the input by following
    # amount: the adjustment
    adjustment = errors*sigmoid_derivative(output)
    
    # finaly update the synaptic_weights for next iteration
    synaptic_weights += np.dot(input_layer.T, adjustment)

print(len(lsoutputs), len(lserrors)) # both are same

# Let us try to plot the progress of loss and accuracy so as to check the flow
import matplotlib.pyplot as plt

# Errors
plt.plot(range(50000),lserrors,label='Errors')
plt.annotate(xy=[100,lserrors[100]],s=lserrors[100],c='r')
plt.annotate(xy=[20000,lserrors[20000]],s=lserrors[20000],c='r')
plt.annotate(xy=[49999,lserrors[-1]],s=lserrors[-1],c='r')
plt.title("Error Change With Each Iteration")
plt.legend()
plt.show()

# Output
plt.plot(range(50000),lsoutputs,label="Outputs")
plt.title("Output Values change with Each Iteration")
plt.legend()
plt.show()

# Both
plt.plot(range(50000),lserrors,label='Errors')
plt.plot(range(50000),lsoutputs,label="Outputs")
plt.title("Errors and Outputs Changes with Iterations")
plt.legend()
plt.show()

# Expected Output
training_op

# Last received Output
output

# If we compare last received output with the Expected output, we see very less difference and hence we can say that with
# each iteration, the neural network adjusted the input so as to get the output with minimum error w.r.t the expected output.
# We cane test with number of iterations that the more the iteration, the more algorithm tends to improve with the accuracy
# by minimizing the errors.
