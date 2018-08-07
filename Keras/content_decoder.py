#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 11:20:49 2018
@author: pabloruizruiz
"""

import os
import numpy as np
import matplotlib.pyplot as plt

import keras.backend as K
from keras.models import Model

from utils import minimize
from utils import VGG_Slice
from utils import load_img_and_preprocess, unpreprocess, scale_

''' Path configurations '''
path_to_images = os.path.join(os.getcwd(), '../Images')
path_to_content = os.path.join(path_to_images, 'content')
path_to_style   = os.path.join(path_to_images, 'style')
path_to_save    = './output_content_decode'


# 1 - Load Content Image

path_to_image = 'content/elephant.jpg'
content_image = load_img_and_preprocess(path_to_image)
h, w = content_image.shape[1:3]

plt.imshow(scale_(unpreprocess(content_image).squeeze(0)))
plt.show()


# 3 - Define the parameters of the image

shape = content_image.shape[1:]
batch_shape = content_image.shape


# 5 - Create the Content Network and the Target Variable

content_net = VGG_Slice(shape, 11)
content_target = K.variable(content_net.predict(content_image))


# 7 - Calculate the loss and gradients respect to input

# 7.1 - Content loss
loss = K.mean(K.square(content_net.output - content_target))
grads = K.gradients(loss, content_net.input)

# 7.4 - Create a callable function that uses the symbolic variables we have created
loss_grads = K.function(inputs = [content_net.input], 
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
plt.savefig(os.path.join(path_to_save,'content_decoded.png'))
plt.show()

plt.plot(history)
plt.savefig('history.png')
plt.show()

