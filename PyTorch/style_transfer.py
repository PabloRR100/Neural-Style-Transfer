#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 17:28:28 2018
@author: pabloruizruiz
"""

# If running from shell
import matplotlib
matplotlib.use('Agg')

import os
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils import image_loader, image_drawer
from utils import gram_matrix, Normalization, get_input_optimizer

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", 'ImportWarning')
warnings.filterwarnings("ignore", 'DeprecationWarning')

from datetime import datetime
now = datetime.now
def time(start):    
    elapsed = (now() - start).total_seconds()
    hours =  int(elapsed/3600)
    minutes = round((elapsed/3600 - hours)*60, 2)
    return hours, minutes


''' Declaration of Images '''

#import argparse
#parser = argparse.ArgumentParser()
#parser.add_argument('-s', '--style', required=True, help="style image source")
#parser.add_argument('-c', '--content', required=True, help="content image source")
#args = vars(parser.parse_args())


''' Configuration Parameters '''

display = False
comments = True
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
gpus = True if torch.cuda.device_count() > 1 else False
mem = False if device == 'cpu' else True

print('Cuda: ', cuda)
print('Device: ', device)
print('GPUs: ', gpus)

imsize = 512 if cuda else 128


''' Configuration of paths '''

#root = os.path.abspath(os.getcwd())
#os.chdir(os.path.join(root, 'PyTorch'))

path_to_images = os.path.join(os.getcwd(), '../Images')
path_to_content = os.path.join(path_to_images, 'content')
path_to_style   = os.path.join(path_to_images, 'styles')
path_to_outputs = os.path.join(path_to_images, 'pytorch_outputs')


assert os.path.exists(path_to_images), 'Image folder does not exist'
assert os.path.exists(path_to_content), 'Content images folder does not exist'
assert os.path.exists(path_to_style), 'Style images folder does not exist'
assert os.path.exists(path_to_outputs), 'Output folder does not exist'

if comments:
    print('Path to : ', path_to_images)
    print('Path to : ', path_to_content)
    print('Path to : ', path_to_style)
    print('Path to : ', path_to_outputs)


# 1 - Load Images

#content_path = os.path.join(path_to_content, args['content'])
content_path = os.path.join(path_to_content, 'ace_pablo.jpg')
content_image = image_loader(content_path, imsize, device)

#style_path = os.path.join(path_to_style, args['style'])
style_path = os.path.join(path_to_style, 'starrynight.jpg')
style_image = image_loader(style_path, imsize, device)

input_image = content_image.clone()
#input_image = torch.randn(content_image.data.size(), device=device)
#plt.figure()
#image_drawer(input_image, title='Input Image')

# Sanity check
assert content_image.size() == style_image.size(), \
    'Content and Image sizes must match'

# Visualize the images
if display:

    plt.figure()
    image_drawer(content_image, title='Content Image')
    
    plt.figure()
    image_drawer(style_image, title='Style Image')


# 2 - Content Loss

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

# 3 - Style Loss

class StyleLoss(nn.Module):

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input



# 4 - Add loss modules in the right order to modules from pretrained model

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']



# 5 - Load Pre-trained Network
''' For style tranfer VGG is the best architecture '''


cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


# Function to create the desired architecture from the pretrained model
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img,
                               content_layers = content_layers,
                               style_layers = style_layers):

    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)
    i = 0  # Control variable to count Conv layers passed
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # Add Content Loss
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        # Add Style Loss
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


# 6 - Training Algorithm
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=2000,
                       style_weight=1000000, content_weight=3):
    """Run the style transfer."""
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)
    
    if gpus: 
        print('Deplying model into gpus...')        
        model = nn.DataParallel(model)
        model.to(device)
    
    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
                
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 200 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # Clip to ensure the values makes sense and are between [0-1]
    input_img.data.clamp_(0, 1)
    return input_img

print('Init Style Transfer Procces')
print('---------------------------')


# 7 - Run and Transer Style!!
#new_title = str(args['content'] + '_' + args['style'])
start = now()
new_title = 'test_outpus'
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_image, style_image, input_image)
print('Time: {} hours {} minutes'.\
      format(time(start)[0], time(start)[1]))

plt.figure()
image_drawer(output, title='Output Image')
plt.savefig(os.path.join(path_to_outputs, new_title))
plt.ioff()
plt.show()
