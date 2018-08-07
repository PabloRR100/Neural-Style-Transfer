#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:40:15 2018
@author: pabloruizruiz

Pipeline to reconstruct the style of an input image
"""


import os
import numpy as np
import matplotlib.pyplot as plt

import keras
import keras.backend as K
import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import MaxPooling2D, AveragePooling2D

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

from utils import minimize
from utils import VGG16_AvgPool, style_Loss
from utils import load_img_and_preprocess, unpreprocess, scale_


''' Path configurations '''
path_to_images = os.path.join(os.getcwd(), '../Images')
path_to_content = os.path.join(path_to_images, 'content')
path_to_style   = os.path.join(path_to_images, 'style')
path_to_save    = './output_style_decode'


# 2 - Load Style Image

path_to_image = 'styles/lesdemoisellesdavignon.jpg'
style_image = load_img_and_preprocess(path_to_image)
plt.imshow(scale_(unpreprocess(style_image).squeeze(0)))


# 3 - Define the parameters of the image

shape = style_image.shape[1:]
batch_shape = style_image.shape


# 6 - Create the Style Network

vgg = VGG16_AvgPool(shape)

# We need to deal with several outputs at different layers of the network
symbolic_conv_outputs = [
  layer.get_output_at(1) for layer in vgg.layers \
  if layer.name.endswith('conv1')
]

style_net = Model(vgg.input, symbolic_conv_outputs)
style_outputs = [K.variable(pred) for pred in style_net.predict(style_image)]


# 7 - Calculate the loss and gradients respect to input

# 7.2 - Style loss
loss = 0
for symbolic, actual in zip(symbolic_conv_outputs, style_outputs):
    loss += style_Loss(symbolic[0], actual[0])

grads = K.gradients(loss, style_net.input)

# 7.4 - Create a callable function that uses the symbolic variables we have created
loss_grads = K.function(inputs = [style_net.input], 
                        outputs = [loss] + grads)


# 7.5 - Wrapper function
def get_loss_and_grads_wrapper(x_vec):
    '''
    Takes a 1D vector, reshape it into an image, calculate loss and gradient
    and flatten each vector back casting into float64
    '''
    l, g = loss_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


final_image, history = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
plt.imshow(scale_(final_image))
plt.savefig(os.path.join(path_to_save,'style_decoded.png'))
plt.show()

plt.plot(history)
plt.savefig(os.path.join(path_to_save,'history.png'))
plt.show()


    
    
    