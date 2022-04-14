# Pix2Pix-GAN

The characteristics of Pix2Pix GAN is the following:
1) The generator receives an image and it outputs a translated
version of the input image.
2) The discriminator is a conditional discriminator which
receives a real or fake image which is conditioned on
the same input image that is received by the generator,
3) The discriminator classifies whether the image is real or
not.
4) The goal of the Pix2Pix GAN is to fool the discriminator,
such that the generator learns how to translate the
images.

The architecture consists of two networks: Generator and
Discriminator. Both the networks are trained such that the
Generator tries to mimic the real data distribution and the
Discriminator network classifies the real images from the
fake images.
• After the training of the generator is complete, the model
is able to output similar images when we input a random
noise as the input.
• The difference between normal and conditional GAN is
that the generator was fed random input conditional on
the label of the input image and then for different types
of random input, it will generate different images from
that class label.

Generator:
The Generator model used in Pix2Pix GAN is called UNet-
Generator model. The model of UnetGenerator is similar to the
encode-decoder architecture with skip connections between
mirrored layers. The characteristics of UnetGenerator is as
follows:
1) The output y is the translated and conditioned version
of x.
2) The skip connections are used to retain the information
that was lost during down-sampling. Also, the skip
connections helps void the vanishing gradient issue in
back propagation.

Discriminator:
The discriminator for Pix2Pix GAN is called PatchGAN,
which produces an output of 30 × 30 matrix. Each cell of
the matrix represents the probability of a 70×70 patch being
real or fake. The model for the PatchGAN architecture is as
follows:
• The discriminator model uses the standard Convolution-
BatchNormalization-ReLU blocks.
• The network will give the output of the shape 30 × 30
which represents whether each 70×70 patch of the input
image is real or fake.
• The value of 0 represents that the image is fake and the
value of 1 represents that the image is real.

![image](https://user-images.githubusercontent.com/70934463/163453815-451919c2-9453-49d0-893e-d6f71f029cbb.png)
Generated streetview images:
![image](https://user-images.githubusercontent.com/70934463/163453857-c34b2f75-bed1-4f4d-8607-b21d47854229.png)
Generated overhead images:
![image](https://user-images.githubusercontent.com/70934463/163454013-eefa6202-d8a8-4e3c-b11f-c337c2154515.png)

