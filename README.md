
<div class=text-justify>
<h1>Neural Style Transfer</h1>

</div>

<div class=text-justify>
Neural style transfer is a computer vision technique consisting on applying the artistic style from one image into the content of another image. This way, the output image will have the intention of the artist who draw the content image, as if the artist who draw the style image painted it.

</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/intro.png?raw=True">


## Index

- Just give me the code  
    - Python Files
        - [PyTorch](https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/PyTorch/style_tranfer.py)
        - [Keras](https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Keras/style_transfer_model.py) 
    - Explained Notebooks
        - [PyTorch Documentation](https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/PyTorch%20Implementation.md)
        - Keras Documentation - To Be Implemented
        
- How it works?
- Reproduce content
- Reproduce style
- Final Picture


<h2> How does it work?</h2>
<hr>
<div class=text-justify>
The publication where neural style transfer was introduced is this: <a href='https://arxiv.org/pdf/1508.06576.pdf'> Arxiv Publication </a>
<br><br>
It is pretty much as applying Convolutional Neural Networks to an image. The trick resides on how the loss of the model is defined. In fact, the weights in NST are fixed, which means we could use a pretrained model and never update its weights
<br><br>
So, what do we maximize? In NST we maximize the output with respect to the input itself!
<br><br>
We know that a CNN allow us to have a representation of the image into what we call feature maps, that can actually be the size we desire. Therefore, what we are looking is for another image that have the convolved outputs as the original images.

</div>


<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/eq1.png?raw=true" alt="eq1" style="width:30%; margin=10px"/>

This would be equivalent to answering the question: what image maximizes these activations?

[intro]: https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/intro.png?raw=true


<h2>Reproduce Content</h2>
<hr>

How could we reproduce the content of an image?

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/reproduce_content.png?raw=true" alt="content"/>


<div class=text-justify>
As mention, we are looking for the result of the convolution to be equivalent. In the figure above, our content image (the elephant) will produce an output after the forward pass on the CNN. We will then change the pixels in out input image (the noisy image) to produce the same output volume.
<br><br>
And compute the loss of this aspect is trivial. We want to know “how far” both volumes are, so we could simple calculate the MSE for the vector resulting of flatten those volumes.
</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/eq2.png?raw=true" alt="eq1" style="width:40%; margin=10px"/>

<u><h3>Note</h3></u>
<div class=text-justify>
We now could think about how CNN works to take more advantage of using it in this task. As we know, CNN propagate representations of the image over the depth dimension, layer-by-layer. This means, that the further we go from the input image, more detail from the image is represented by the output of that particular layer.
    <br><br>
Since right now we want to capture the content of the image, we are not actually interested in using the entire network, since we will lose precision in the content encoding. Therefore, the above equations will be more accurate by changing:
</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/eq3.png?raw=true" alt="eq1" style="width:40%; margin=10px"/>

<div class=text-justify>
The next picture shows how out prediction image looks if we convolve the content image until the last convolutional layer. We can see how the main features are still kept, and we have “made room” for the style matching. 
Note that if we recreate after passing the content image only until the first convolution, it will be very sharp, close to the actual content image. So, the content will be kept, but there wouldn’t be that space for the style.
</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/content_decoded.png?raw=true" alt="content_decoded" style="width:40%; margin=10px"/>


<h2>Reproduce Style</h2>
<hr>

<div class=text-justify>
We have already cover how are we going to attempt to replicate the content of a target image. But, at the same time we keep track of replicating the style of the style image. In fact, if we were only to replicate the output using the input image, why did we bother to use a CNN in the first place?
<br><br>
As we saw before, the deeper we go into the CNN the more detail we lose. However, this is the key point for extracting the style of an image! We want to capture the style, we are not looking at the details at pixel level, we are looking for more general representations of the image.
</div>

<u><h3>Gram Matrix</h3></u>
<div class=text-justify>
We need to explain the concept of the gram matrix in order to understand how to reproduce the style.
</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/eq4.png?raw=true" alt="content" style="width:15%; margin=30px"/>

<div class=text-justify>
The gram matrix G is computed by multiplying the input image by its transpose. The intuition behind this matrix is that it contains non-localized information about the images, such as textures and other features that conform the style of an artist. The gram matrix formulation suggest that it is computing the <b><i>autocorrelation</i></b>. This means, how correlated are the input with itself.
    <br><br>
But our convolution output is a 3D volume. We need to do a transformation to be able to convert it to a 2D matrix. Therefore, we flatten the volume along the spatial dimension.

</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/gram_matrix.png?raw=true" alt="content"/>


<div class=text-justify>
The resulting gran matrix becomes size (Features x Features). Interestingly, the image size dimensions disappear. This agrees with the statement we mention before. When it comes to texture, we don’t care about the localization on the image, it must be consistent at every location on the image.
    <br><br>
In the original paper, they took the output at 5 different locations of the VGG:
</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/style_loss.png?raw=true" alt="content" style="width:85%"/>

By doing this, it allows us to capture patterns of different sizes, since they are at different depth of the CNN. So, to compute the final loss of the style, we could simply make a weighted average of the errors at every defined output convolution. 

<u><h3>Note</h3></u>
<div class=text-justify>
These are one of the hyperparameters we have to decide for the architecture of our style transfer model. The other ones are regarding how much importance we give to the content being well reproduced against the style, to compute the final global loss function. 
</div>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/eq5.png?raw=true" alt="content" style="width:25%"/>

<h1>Final Picture</h1>
<hr>

<img src="https://github.com/PabloRR100/Neural-Style-Transfer/blob/master/Documentation/images/global_picture.png?raw=true" alt="content"/>

# Credits
---

Thanks to :

- 'The Lazy Programmer' for his wonderful [course on Advanced Computer Vision](https://www.udemy.com/advanced-computer-vision/) on Udemy

- Alexis Jacq for his outstanding tutorial on Neural Style Tranfer on [PyTorch Official Tutorials](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)
