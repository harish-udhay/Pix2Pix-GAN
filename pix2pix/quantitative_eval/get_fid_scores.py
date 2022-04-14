from generator import Generator

import torch
import os, shutil, json, cv2, numpy as np
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils as img_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "./out/"
gen_weights_dir = os.path.join(base_dir, "generator")
generator = Generator(3, 3, 64, use_dropout=True).cuda().float()
real_image_dir = os.path.join(base_dir, "real_images")
fakes_dir = ""
processed_data_dir = ""

def split_image(image):
    image = np.array(image)
    width = image.shape[1]
    width_half = width // 2
    input_image = image[:, :width_half, :]
    target_image = image[:, width_half:, :]
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
    cropped_image = random_crop(stacked_image, dim=[256, 256, 3])
    
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
        inp, tar = split_image(image)
        inp, tar = random_jittering_mirroring(inp, tar)
        inp, tar = normalize(inp, tar)
        image_a = torch.from_numpy(inp.copy().transpose((2,0,1)))
        image_b = torch.from_numpy(tar.copy().transpose((2,0,1)))
        return image_a, image_b

def process_data(path, batch_size = 1):
        training_data = datasets.ImageFolder(
            root = path,    
            transform = transforms.Compose([Preprocess()])
        )
        data_loader = DataLoader(  training_data,
                            batch_size = batch_size, 
                            shuffle = True)

        aerial_dir = os.path.join(real_image_dir, "aerial", "1")
        street_dir = os.path.join(real_image_dir, "street")

        if not os.path.exists(aerial_dir):
            os.makedirs(aerial_dir)
        if not os.path.exists(street_dir):
            os.makedirs(street_dir)

        for i, data in enumerate(data_loader):
            img_utils.save_image(data[0][0], os.path.join(aerial_dir, "real_" + str(i) + ".png"))
            img_utils.save_image(data[0][1], os.path.join(street_dir, "real_" + str(i) + ".png"))

# Generate fakes by loading saved generator models
def inference(num_fakes, data_loader):
    fakes_dir = os.path.join(base_dir, "generated_fakes")
    if not os.path.exists(fakes_dir):
        os.makedirs(fakes_dir)
    for i in range(num_fakes):
        fake_img = generator([data for j, data in enumerate(data_loader) if i == j][0][0].to(device))
        img_utils.save_image(fake_img, os.path.join(fakes_dir, "fake_" + str(i) + ".png"))

# Calculate FID score
def get_fid_score(paths, batch_size):
    return calculate_fid_given_paths(paths, batch_size, device, dims = 2048)

if __name__ == "__main__":
    process_data(real_image_dir)

    fid_scores = {}

    # Iteratively load saved generator models and use them to generate fakes.
    # Subsequently, compare the generated fakes to test data and compute FID score
    for _, _, files in os.walk(gen_weights_dir):
        for weight_file in files:
            model = torch.load(os.path.join(gen_weights_dir, weight_file))
            generator.load_state_dict(model.get("Gen_Model"), strict = False)
            num_files = 100
            inference_data = datasets.ImageFolder( root = os.path.join(real_image_dir, "aerial"), transform = transforms.ToTensor())
            data_loader = DataLoader(inference_data, batch_size = 1)
            inference(num_fakes = num_files, data_loader = data_loader)
            paths = [os.path.join(base_dir, "generated_fakes"), 
                    os.path.join(real_image_dir, "street")]
            batch_size = min(50, num_files)
            score = get_fid_score(paths, batch_size)
            epoch = int(weight_file.split("_")[2][:-4])
            print("Processing Epoch: {}".format(epoch))
            fid_scores[epoch] = score
    
    with open(os.path.join(base_dir, "fid.json"), "w") as f:
        json.dump(fid_scores, f)

    shutil.rmtree(os.path.join(real_image_dir, "aerial"))
    shutil.rmtree(os.path.join(real_image_dir, "street"))
        