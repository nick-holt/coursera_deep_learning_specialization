# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:31:11 2018

@author: Nick Holt
"""

### Hello World
test = "Hello World!"

print("test: " + test)

### basic_sigmoid function using math library
import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    s = 1/(1+math.exp(-x))
    
    return s

basic_sigmoid(3)

### basic sigmoid using numpy

### One reason why we use "numpy" instead of "math" in Deep Learning ###
x = [1, 2, 3]
basic_sigmoid(x) # you will see this give an error when you run it, because x is a vector.

import numpy as np

# example of np.exp
x = np.array([1, 2, 3])
print(np.exp(x)) # result is (exp(1), exp(2), exp(3))

# using numpy.exp
def sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    s = 1/(1+np.exp(-x))
    
    return s

basic_sigmoid(3)

x = np.array([1, 2, 3])
sigmoid(x)

# First Gradient Function: Sigmoid Derivative
def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    s = sigmoid(x)
    ds = s*(1-s)
    
    return ds

x = np.array([1, 2, 3])
print ("sigmoid_derivative(x) = " + str(sigmoid_derivative(x)))

# Reshaping Arrays

# image2vector function
def image2vector(image):
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    
    v = image.reshape((image.shape[0]*image.shape[1]*image.shape[2])) # v.shape[0] = a ; v.shape[1] = b ; v.shape[2] = c
    
    return v

# This is a 3 by 3 by 2 array, typically images will be (num_px_x, num_px_y,3) where 3 represents the RGB values
image = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])
    
print(str(image2vector(image)))

# normalizeRows Function
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis = 1, keepdims = True)
    
    # Divide x by its norm.
    x = x/x_norm

    return x

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
    
print("normalizeRows(x) = " + str(normalizeRows(x)))

# broadcasting and the softmax function
def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x, axis = 1, keepdims = True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum
    
    return s

x = np.array([
    [9, 2, 5, 0, 0],
    [7, 5, 0, 0 ,0]])

print("softmax(x) = " + str(softmax(x)))









