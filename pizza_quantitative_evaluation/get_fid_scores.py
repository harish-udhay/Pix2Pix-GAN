from generator import Generator

import torch
import os, shutil, json
from pytorch_fid.fid_score import calculate_fid_given_paths
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, utils as img_utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_dir = "./out-synthetic/"
gen_weights_dir = os.path.join(base_dir, "generator")
generator = Generator().to(device)
real_image_dir = os.path.join(base_dir, "real_images")
fakes_dir = ""
processed_data_dir = ""

def process_data(path, dimensions, batch_size = 1):
        transform = transforms.Compose([
            transforms.Resize(dimensions),
            transforms.CenterCrop(dimensions),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        training_data = datasets.ImageFolder(
            root = path,    
            transform = transform
        )
        data_loader = DataLoader(  training_data,
                            batch_size = batch_size, 
                            shuffle = True)
        processed_data_dir = os.path.join(real_image_dir, "processed")

        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
        for i, data in enumerate(data_loader):
            img_utils.save_image(data[0], os.path.join(processed_data_dir, "real_" + str(i) + ".png"))

def inference(num_fakes = 50):
    fakes_dir = os.path.join(base_dir, "generated_fakes")
    if not os.path.exists(fakes_dir):
        os.makedirs(fakes_dir)
    for i in range(num_fakes):
            noise = torch.randn(1, 3, 256, 256, device = device)
            fake_img = generator(noise)
            img_utils.save_image(fake_img, os.path.join(fakes_dir, "fake_" + str(i) + ".png"))

def get_fid_score(paths, batch_size):
    return calculate_fid_given_paths(paths, batch_size, device, dims = 2048)

if __name__ == "__main__":
    process_data(real_image_dir, 256)

    fid_scores = {}

    for _, _, files in os.walk(gen_weights_dir):
        for weight_file in files:
            generator.load_state_dict(torch.load(os.path.join(gen_weights_dir, weight_file)))
            num_files = 100
            inference(num_fakes = num_files)
            paths = [os.path.join(base_dir, "generated_fakes"), 
                    os.path.join(real_image_dir, "processed")]
            batch_size = min(50, num_files)
            score = get_fid_score(paths, batch_size)
            epoch = int(weight_file.split("_")[2][:-4])
            print("Processing Epoch: {}".format(epoch))
            fid_scores[epoch] = score
    
    with open(os.path.join(base_dir, "fid.json"), "w") as f:
        json.dump(fid_scores, f)

    shutil.rmtree(os.path.join(real_image_dir, "processed"))
        