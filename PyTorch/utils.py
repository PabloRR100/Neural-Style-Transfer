#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 18:46:39 2018
@author: pabloruizruiz
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:28:28 2018
@author: pabloruizruiz
"""

import os
import copy
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms


# 1 - Function to load and preprocess images
def image_loader(image, imsize, device):
    
    path = os.path.exists(image)
    print(image)
    assert path, 'Path to images not valid'
    
    # Define transformation
    loader = transforms.Compose([
        transforms.Resize((imsize, imsize)),
        transforms.ToTensor()])
    
    image = Image.open(image)       # Open image
    image = loader(image)           # Apply preprocessing
    image = image.unsqueeze(0)      # Include batch dimension
    if image.size()[1] == 4:
        image = image[:,:3,:,:]
    image = image.to(device, torch.float)
    return image


# 2 - Function to plot images
unloader = transforms.ToPILImage()  # Reconvert into PIL image
plt.ion()

def image_drawer(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated


# 3 - Function to calculate Gram Matrix
def gram_matrix(input):
    a, b, c, d = input.size()  #[batch_size, feature_maps, feat_map_dimensions]
    features = input.view(a * b, c * d)     # Reshape
    G = torch.mm(features, features.t())    # Calculate Gram Matrix
    return G.div(a * b * c * d) 


# 4 - Class to create normalizers to create modules with Sequential()
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std
    

# 5 - Function to create the optimizer with respect to the inputs
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer
    
    

