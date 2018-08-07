#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:20:49 2018
@author: pabloruizruiz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import keras
import tensorflow as tf
print('Keras version: ', keras.__version__)
print('Tensorflow version: ', tf.__version__)

import keras.backend as K
from keras.models import Model

from utils import minimize
from utils import VGG16_AvgPool, style_Loss
from utils import load_img_and_preprocess, unpreprocess, scale_


''' Path configurations '''
path_to_images = os.path.join(os.getcwd(), '../Images')
path_to_content = os.path.join(path_to_images, 'content')
path_to_style   = os.path.join(path_to_images, 'styles')
path_to_save    = './outputs'


# 1 - Load Contnt Image

path_to_image = os.path.join(path_to_content, 'elephant.jpg')
content_image = load_img_and_preprocess(path_to_image)
plt.imshow(scale_(unpreprocess(content_image).squeeze(0)))
img_shape = content_image.shape[1:3]

# 2 - Load Style Image

path_to_image = os.path.join(path_to_style, 'starrynight.jpg')
style_image = load_img_and_preprocess(path_to_image, shape=img_shape)
plt.imshow(scale_(unpreprocess(style_image).squeeze(0)))


# 3 - Define the parameters of the image

shape = content_image.shape[1:]
batch_shape = content_image.shape


# 4 - Create the VGG Network

vgg = VGG16_AvgPool(shape)
vgg.summary()


# 5 - Create the Content Network

content_net = Model(vgg.input, vgg.layers[13].get_output_at(0))
content_target = K.variable(content_net.predict(content_image))


# 6 - Create the Style Network

# We need to deal with several outputs at different layers of the network
symbolic_conv_outputs = [
  layer.get_output_at(1) for layer in vgg.layers \
  if layer.name.endswith('conv1')
]

style_net = Model(vgg.input, symbolic_conv_outputs)
style_outputs = [K.variable(pred) for pred in style_net.predict(style_image)]
style_W = [0.2,0.4,0.3,0.5,0.2]      # Empirical values ???


# 7 - Calculate each loss and wrap them into a unique loss

# 7.1 - Content loss
loss = K.mean(K.square(content_net.output - content_target))

# 7.2 - Style loss
for w, symbolic, actual in zip(style_W, symbolic_conv_outputs, style_outputs):
    loss += style_Loss(symbolic[0], actual[0])
    
# 7.3 - Calculate the gradients respect to the inputs 
grads = K.gradients(loss, vgg.input)

# 7.4 - Create a callable function that uses the symbolic variables we have created
loss_grads = K.function(inputs = [vgg.input], 
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
plt.savefig(os.path.join(path_to_save,'output.png'))
plt.show()

plt.plot(history)
plt.savefig(os.path.join(path_to_save,'history.png'))
plt.show()









