
# PyTorch Implementation


In this notebook it is explained the steps to follow on PyTorch.  
Note that the real script is in the PyTorch folder. Therefore, in order for this to run, you need to take care of the dependencies references to be able to import them corretly


### Import dependecies


```python
import os
import copy
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PyTorch.utils import image_loader, image_drawer
from PyTorch.utils import gram_matrix, Normalization, get_input_optimizer
```

### Configuration

We need to define the path to the images.  
Remember from the documentation that we need images to replicate their content and style image to copy their artistic styles.  
We also want to make a folder to store the outputs.


```python
''' Configuration Parameters '''
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
os.chdir('/Users/pabloruizruiz/OneDrive/Proyectos/AI/ML/COMPUTER VISION/Neural-Style-Transfer/PyTorch')

#imsize = 512 if cuda else 128
imsize = 512
path_to_images = os.path.join(os.getcwd(), '../Images')
path_to_content = os.path.join(path_to_images, 'content')
path_to_style   = os.path.join(path_to_images, 'styles')
path_to_outputs = os.path.join(path_to_images, 'pytorch_outputs')


assert os.path.exists(path_to_images), 'Image folder does not exist'
assert os.path.exists(path_to_content), 'Content images folder does not exist'
assert os.path.exists(path_to_style), 'Style images folder does not exist'
assert os.path.exists(path_to_outputs), 'Output folder does not exist'
```

### 1 - Load images

The starting point could be a noise image (like the shown in the documentation) or also the content image. So the content loss will be small and the style loss will be big at the beggining, until they get balanced depending on th weights we define for them.


```python
# 1 - Load Images

content_path = os.path.join(path_to_content, 'retiro.jpg')
content_image = image_loader(content_path, imsize, device)

style_path = os.path.join(path_to_style, 'fornite_map.jpg')
style_image = image_loader(style_path, imsize, device)

# To start with the content image
input_image = content_image.clone()  
# To start with a noise image
# input_image = torch.randn(content_image.data.size(), device=device)

# Sanity check
assert content_image.size() == style_image.size(), \
    'Content and Image sizes must match'
```


```python
# Visualize the images
plt.figure()
image_drawer(content_image, title='Content Image')

plt.figure()
image_drawer(style_image, title='Style Image')
```

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/content_image.png?raw=true" alt="content" style="width:60%"/>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/style_image.png?raw=true" alt="style" style="width:60%"/>

### 2 - Define the losses

The Content Loss is simply the distance of the image with respect to the target content image.  

The Style Loss is the weighted average distance of the Gram Matrix of the feature maps at different layers of the network with the Gram Matrix of the style target.


```python
# 2 - Content Loss
class ContentLoss(nn.Module):

    def __init__(self, target):
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
```

### 4 - Define the layers where we will extract the feature maps

As we said, we will average the feature maps from different convolutional layers of the CNN for the style loss.  

But also, for the content we will get the feature maps at some layer. As you could see in the documentation, we get the content from some late convolution layer, as we are interested in keeping **only the most important feature** and leave some space to capture the style


```python
# 4 - Add loss modules in the right order to modules from pretrained model

content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


```

### 5 - Load a pretrained Network

In Neural Style Transfer, the weights of the neural network remains constant.  
This is because the gradients are computed respect to the input pixels, not any wieghts.  
Therefore, it makes sense to make use of a powerful already pre-trained model like VGG19, in this particular case:


```python
# 5 - Load Pre-trained Network
''' For style tranfer VGG is the best architecture '''

cnn = models.vgg19(pretrained=True).features.to(device).eval()

cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
```

We are keeping only the Sequential module ```features``` as we are only interested in this part of the VGG - we are getting rid of the classifier block.  

Also, as we are not training the network, we set in into evaluation mode with ```.eval()```

Furthermore, as VGG is trained on images with the channels normalized, we used the values used on that training to normlize our images before feeding them to the VGG

#### Construct the model from this pretrained model

Now we are just going to replicate the architecture of the pretrained Sequential modules of the VGG, and append our defined losses on the layer we chose above


```python
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
```

### 6 - Training Algorithm

The original paper suggest to use ```L-BFGS```algorithm to run the gradient descense optimization.



```python
# 6 - Training Algorithm
def run_style_transfer(cnn, normalization_mean, normalization_std,
                       content_img, style_img, input_img, num_steps=2000,
                       style_weight=1000000, content_weight=10):
    print('Building model...')
    model, style_losses, content_losses = get_style_model_and_losses(cnn,
        normalization_mean, normalization_std, style_img, content_img)
    optimizer = get_input_optimizer(input_img)

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
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # Clip to ensure the values makes sense and are between [0-1]
    input_img.data.clamp_(0, 1)
    return input_img

```

### 7 - Run and fun


```python
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_image, style_image, input_image)

plt.figure()
image_drawer(output, title='Output Image')
plt.savefig(os.path.join(path_to_outputs, 'Retiro2'))
plt.ioff()
plt.show()
```

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Images/pytorch_outputs/retiro_fornite_clean.png?raw=True" alt="content" style="width:40%"/>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/intro_retiro.png?raw=True" alt="content" style="width:100%"/>


Some previous results that was not that good, so you can see how much the results can be changed depending on the choice of the hyperparamaters:

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Images/pytorch_outputs/retiro_fornite_old.png?raw=True" alt="content" style="width:40%"/>
