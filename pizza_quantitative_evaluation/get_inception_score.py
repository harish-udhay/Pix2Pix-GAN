from generator import Generator

import torch
import os, json

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from torchvision import utils as img_utils

from matplotlib.image import imread

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Set the base dir where the data is kept
base_dir = "./out-synthetic/"
gen_weights_dir = os.path.join(base_dir, "generator")
generator = Generator().to(device)
fakes_dir = ""

# Function to generate fakes using the generator
def inference(num_fakes = 50):
    img_array = []
    fakes_dir = os.path.join(base_dir, "generated_fakes")
    if not os.path.exists(fakes_dir):
        os.makedirs(fakes_dir)
    for i in range(num_fakes):
            noise = torch.randn(1, 3, 256, 256, device = device)
            fake_img = generator(noise)
            img_utils.save_image(fake_img, os.path.join(fakes_dir, "fake_" + str(i) + ".png"))
    for _, _, files in os.walk(fakes_dir):
        for img in files:
            image = imread(os.path.join(fakes_dir, img))
            image = np.moveaxis(image, -1, 0)
            img_array.append(image)
    return np.array(img_array)

# Function to calculate inception score
def get_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == "__main__":
    # Dict to save all inception scores
    inception_scores = {}

    for _, _, files in os.walk(gen_weights_dir):
        for weight_file in files:
            generator.load_state_dict(torch.load(os.path.join(gen_weights_dir, weight_file)))
            num_files = 300
            img_array = inference(num_fakes = num_files)
            score = get_inception_score(img_array, resize = True)
            epoch = int(weight_file.split("_")[2][:-4])
            print("Processing Epoch: {}".format(epoch))
            inception_scores[epoch] = score
    
    with open(os.path.join(base_dir, "is.json"), "w") as f:
        json.dump(inception_scores, f)
