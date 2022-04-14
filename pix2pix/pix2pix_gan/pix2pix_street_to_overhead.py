import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from datetime import date
import cv2
import os
import os.path
import pickle
import torchvision
from torch.autograd import Variable
import random
import pandas as pd
import argparse

def read_image(image):
    image = np.array(image)
    width = image.shape[1]
    width_half = width // 2
    target_image = image[:, :width_half, :]
    input_image = image[:, width_half:, :]
    input_image = input_image.astype(np.float32)
    target_image = target_image.astype(np.float32)
    return input_image, target_image

def random_crop(image, dim):
    height, width, _ = dim
    x, y = np.random.uniform(low=0,high=int(height-256)), np.random.uniform(low=0,high=int(width-256))  
    return image[:, int(x):int(x)+256, int(y):int(y)+256]

    

def random_jittering_mirroring(input_image, target_image, height=286, width=286):
    
    #resizing to 286x286
    input_image = cv2.resize(input_image, (height, width) ,interpolation=cv2.INTER_NEAREST)
    target_image = cv2.resize(target_image, (height, width),
                               interpolation=cv2.INTER_NEAREST)
    
    #cropping (random jittering) to 256x256
    stacked_image = np.stack([input_image, target_image], axis=0)
    cropped_image = random_crop(stacked_image, dim=[IMG_HEIGHT, IMG_WIDTH, 3])
    
    input_image, target_image = cropped_image[0], cropped_image[1]
    #print(input_image.shape)
    if torch.rand(()) > 0.5:
     # random mirroring
        input_image = np.fliplr(input_image)
        target_image = np.fliplr(target_image)
    return input_image, target_image

def normalize(inp, tar):
    input_image = (inp / 127.5) - 1
    target_image = (tar / 127.5) - 1
    return input_image, target_image

class Preprocess(object):
    def __call__(self, image):
        inp, tar = read_image(image)
        inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))
        image_b = torch.from_numpy(tar.copy().transpose((2,0,1)))
        return image_a, image_b
    
DIR = '/common/home/hu33/pytorch_implementation/stitched_images/train/'
DIR_VAL = '/common/home/hu33/pytorch_implementation/stitched_images/val/'
n_gpus = 4
batch_size = 6
IMG_HEIGHT =IMG_WIDTH = 256
global_batch_size = batch_size
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)


train_ds = dset.ImageFolder(DIR, transform=transforms.Compose([
        Preprocess()]))
train_dl = torch.utils.data.DataLoader(train_ds, global_batch_size)

val_ds = dset.ImageFolder(DIR_VAL, transform=transforms.Compose([
        Preprocess()]))

val_dl  = torch.utils.data.DataLoader(val_ds, global_batch_size)
device = torch.device("cuda:0" if (torch.cuda.is_available() and n_gpus > 0) else "cpu")

def init_weights(model):
    def init_func(m):  # define the initialization function
        cn = m.__class__.__name__
        if hasattr(m, 'weight') and (cn.find('Conv')) != -1:
            torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif cn.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    model.apply(init_func)
    
class Generator(nn.Module):
    def __init__(self, input_channels, output_channels, num_filters=64, use_dropout=False):
        super(Generator, self).__init__()

        ub1 = SkipConnection(num_filters * 8, num_filters * 8, innermost_layer=True, input_channels=None, submodule=None, norm_layer=nn.BatchNorm2d) 

        ub2 = SkipConnection(num_filters * 8, num_filters * 8, input_channels=None, submodule=ub1, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        ub3 = SkipConnection(num_filters * 8, num_filters * 8, input_channels=None, submodule=ub2, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)
        ub4 = SkipConnection(num_filters * 8, num_filters * 8, input_channels=None, submodule=ub3, norm_layer=nn.BatchNorm2d, use_dropout=use_dropout)

        ub5 = SkipConnection(num_filters * 4, num_filters * 8, input_channels=None, submodule=ub4, norm_layer=nn.BatchNorm2d)
        ub6 = SkipConnection(num_filters * 2, num_filters * 4, input_channels=None, submodule=ub5, norm_layer=nn.BatchNorm2d)
        ub7 = SkipConnection(num_filters, num_filters * 2, input_channels=None, submodule=ub6, norm_layer=nn.BatchNorm2d)
        
        self.model = SkipConnection(output_channels, num_filters,outermost_layer=True, input_channels=input_channels, submodule=ub7, norm_layer=nn.BatchNorm2d)  

    def forward(self, input):
        return self.model(input)

class SkipConnection(nn.Module):
    def __init__(self, output_channels, inner_channels, input_channels=None, submodule=None, outermost_layer=False, innermost_layer=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SkipConnection, self).__init__()
        self.outermost_layer = outermost_layer
        if input_channels is None:
            input_channels = output_channels
        down_sampling_conv = nn.Conv2d(input_channels, inner_channels, kernel_size=4,
                             stride=2, padding=1, bias=False)
        down_sampling_relu = nn.LeakyReLU(0.2, True)
        down_sampling_norm = norm_layer(inner_channels)
        up_sampling_relu = nn.ReLU(True)
        up_sampling_norm = norm_layer(output_channels)

        if outermost_layer:
            up_sampling_conv = nn.ConvTranspose2d(inner_channels * 2, output_channels, kernel_size=4, stride=2, padding=1)
            downSample = [down_sampling_conv]
            upSample = [up_sampling_relu, up_sampling_conv, nn.Tanh()]
            model = downSample + [submodule] + upSample
        
        elif innermost_layer:
            up_sampling_conv = nn.ConvTranspose2d(inner_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=False)
            downSample = [down_sampling_relu, down_sampling_conv]
            upSample = [up_sampling_relu, up_sampling_conv, up_sampling_norm]
            model = downSample + upSample
        
        else:
            up_sampling_conv = nn.ConvTranspose2d(inner_channels * 2, output_channels, kernel_size=4, stride=2, padding=1, bias=False)
            downSample = [down_sampling_relu, down_sampling_conv, down_sampling_norm]
            upSample = [up_sampling_relu, up_sampling_conv, up_sampling_norm]

            if use_dropout:
                model = downSample + [submodule] + upSample + [nn.Dropout(0.5)]
            else:
                model = downSample + [submodule] + upSample

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.outermost_layer:
            return self.model(input)
        else:   # add skip connections
            return torch.cat([input, self.model(input)], 1)

class Discriminator(nn.Module):
    def __init__(self, input_channel, num_filters=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(Discriminator, self).__init__()
        sequence = [nn.Conv2d(input_channel, num_filters, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, True)]
        op_increment = 1
        inp_increment = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            inp_increment = op_increment
            op_increment = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(num_filters * inp_increment, num_filters * op_increment, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(num_filters * op_increment),
                nn.LeakyReLU(0.2, True)
            ]

        inp_increment = op_increment
        op_increment = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(num_filters * inp_increment, num_filters * op_increment, kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(num_filters * op_increment),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(num_filters * op_increment, 1, kernel_size=4, stride=1, padding=1), nn.Sigmoid()]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser('resume epoch')
parser.add_argument(
    "--resume",
    dest = "resume",
    action='store_true',
    # default=False,
    help = "Resume training from a previous checkpoint"
)
parser.add_argument(
    "--checkpoint",
    dest = "checkpoint_path",
    type = str,
    help = "Specify path to checkpoint"
)
args = parser.parse_args()

if args.resume:
    checkpoint = torch.load(args.checkpoint_path)
    epoch = checkpoint['Epoch'] + 1
    print(epoch)

    generator = Generator(3, 3, 64, use_dropout=True).cuda().float()
    generator.load_state_dict(checkpoint['Gen_Model'], strict=False)
    generator = torch.nn.DataParallel(generator)

    discriminator = Discriminator(6).cuda().float()
    discriminator.load_state_dict(checkpoint['Disc_Model'], strict=False)
    discriminator = torch.nn.DataParallel(discriminator)

    
else:
    generator = Generator(3, 3, 64, use_dropout=True).cuda().float()
    init_weights(generator)
    generator = torch.nn.DataParallel(generator)  # multi-GPUs

    discriminator = Discriminator(6).cuda().float()
    init_weights(discriminator)
    discriminator = torch.nn.DataParallel(discriminator)
    epoch = 0

adversarial_loss = nn.BCELoss() 
l1_loss = nn.L1Loss()

def generator_loss(generated_image, target_img, G, real_target):
    gen_loss = adversarial_loss(G, real_target)
    l1_l = l1_loss(generated_image, target_img)
    gen_total_loss = gen_loss + (100 * l1_l)
    #print(gen_loss)
    return gen_total_loss

def discriminator_loss(output, label):
    disc_loss = adversarial_loss(output, label)
    return disc_loss

EPOCHS = 15
D_losses = []
G_losses = []

lr = 2e-4
beta1 =0.5
D_optimizer = optim.Adam(discriminator.parameters(),lr=lr,betas=(beta1,0.999))
G_optimizer = optim.Adam(generator.parameters(),lr=lr,betas=(beta1,0.999))

from tqdm import tqdm
for i in tqdm(range(epoch, EPOCHS)): 
    for (input_img, target_img), _ in tqdm(train_dl,leave=False):
       
        real_label = Variable(torch.ones(input_img.size(0), 1, 30, 30).to(device))
        fake_label = Variable(torch.zeros(input_img.size(0), 1, 30, 30).to(device))
        
        D_optimizer.zero_grad()
        input_img = input_img.to(device)
        target_img = target_img.to(device)
        
        # generator forward pass
        generated_img = generator(input_img)
        
        # train discriminator with fake/generated images
        disc_inp = torch.cat((input_img, generated_img), 1)
        D_out_fake = discriminator(disc_inp.detach())
        D_fake_loss = discriminator_loss(D_out_fake, fake_label)
        
        # train discriminator with real images
        disc_inp = torch.cat((input_img, target_img), 1)                   
        D_out_real = discriminator(disc_inp)
        D_real_loss = discriminator_loss(D_out_real,  real_label)
    
        # average discriminator loss
        D_loss = (D_real_loss + D_fake_loss)/2
        D_losses.append(D_loss)
        # compute gradients and run optimizer step
        D_loss.backward()
        D_optimizer.step()
        
        
        # Train generator with real labels
        G_optimizer.zero_grad()
        inp_gen_img = torch.cat((input_img, generated_img), 1)
        D_out = discriminator(inp_gen_img)
        G_loss = generator_loss(generated_img, target_img, D_out, real_label)                                 
        G_losses.append(G_loss)
        # compute gradients and run optimizer step
        G_loss.backward()
        G_optimizer.step()
        
        
    if i%1 ==0:
        save_dict= {}
        save_dict["Gen_Model"] = generator.state_dict()
        save_dict["Disc_Model"] = discriminator.state_dict()
        save_dict["Epoch"]= i
        print()
        torch.save(save_dict,f"weights/i2i_epoch_2_{i+1}.pth")
    
    
    for (inputs, targets), _ in val_dl:
        inputs = inputs.to(device)
        targets = targets.to(device)
        generated_output =  generator(inputs)
        super_img = [inputs, targets, generated_output]
        super_img_cat = torch.cat(super_img, 0)
        vutils.save_image(generated_output.data[:10], "/common/home/hu33/pytorch_implementation/generated_images/"+'sample_2_%d'%(i+1) + '.png', nrow=5, normalize=True)
        # grid = vutils.make_grid(super_img)
        vutils.save_image(super_img_cat, "/common/home/hu33/pytorch_implementation/generated_images/grids/"+'sample_2_%d'%(i+1) + '.png', nrow=5, normalize=True)
    try:
        #print(type(G_losses), G_losses)
        #G_losses =  np.asarray(G_losses.cpu())
        #print("------------\n",G_losses)
        G_df = pd.DataFrame(G_losses)
        G_df.to_csv('G_losses.csv')
        D_df = pd.DataFrame(D_losses)
        D_df.to_csv('D_losses.csv')
    except Exception as err:
        print(err)
