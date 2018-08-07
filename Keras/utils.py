#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 15:17:20 2018
@author: pabloruizruiz
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b

import keras.backend as K
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

now = datetime.now


# GENERAL HELPER FUNCTIONS
# ------------------------

# 1 - Function to adapt the image to matplotlib

def unpreprocess(img):
    ''' Converts to RGB '''
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


# 2 - Scale image pixels
    
def scale_(p):
    return (p - p.min()) / p.max()


# 3 - Function to load images

def load_img_and_preprocess(path, shape=None):
   img = image.load_img(path, target_size=shape) 
   x = image.img_to_array(img)      # load_img returns an Image object 
   x = np.expand_dims(x, axis=0)    # Keras expects batch dimension
   return preprocess_input(x)


# 3 - Define the Optimization steps
def minimize(fn, epochs, batch_size):
    
    history = []
    x = np.random.randn(np.prod(batch_size)) # Initial guess
    
    start = now()
    for i in range(epochs):
        
        x, l, _ = fmin_l_bfgs_b(func=fn, x0=x, maxfun=20)
        x = np.clip(x, -127, 127)
        history.append(l)
        print('Iter: {}. Loss: {}.'.format(i, l))
    
    print('Duration: ', now() - start)
    new_image = x.reshape(*batch_size)
    new_image = unpreprocess(new_image)
    return new_image[0], history




# CONTENT HELPER FUNCTIONS
# ------------------------
    
# 1 - Change the MaxPool layers of VGG for AvgPool (better performance)

def VGG16_AvgPool(shape):
    
    # Brings default pre-trained VGG 
    vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
    
    # Create VGG by-layer
    net = Sequential()
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            net.add(AveragePooling2D())
        else:
            net.add(layer)
    return net


# 2 - Function to make partial models from VGG

def VGG_Slice(shape, N):

    assert N > 1 and N < 13, 'VGG is between [1,13] blocks long'
    
    # Append until Conv blocks have been reached
    vgg = VGG16_AvgPool(shape)
    net = Sequential()
    n = 0
    for layer in vgg.layers:
        if layer.__class__ == Conv2D:
            n += 1 
        net.add(layer)
        if n >= N:
            break
    return net




# STYLE HELPER FUNCTIONS
# ----------------------
    
# 1 - Function to compute the gram matrix

def gram_matrix(img):
    
    X = K.permute_dimensions(img, (2,0,1))  # Channel dimension first 
    X = K.batch_flatten(X)                  # C * (H,W) 
    
    N = img.get_shape().num_elements()
    G = (1 / N) * K.dot(X, K.transpose(X))
    return G


# 2 - Define the Loss for the Style
    
def style_Loss(output, target):
    
    error = gram_matrix(output) - gram_matrix(target)
    return K.mean(K.square(error)) # Mean Squared Error - MSE
