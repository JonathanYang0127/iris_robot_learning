import torch
import numpy as np
from rlkit.torch import pytorch_util as ptu
import os
import glob
import pickle as pkl

ROOT = '/2tb/home/patrickhaoy/data/affordances/data/reset_free_v5_rotated_top_drawer/'
NEW_ROOT = '/2tb/home/patrickhaoy/data/affordances/data/reconstructed_reset_free_v5_rotated_top_drawer/'
local_path = ROOT + 'best_vqvae.pt'
all_files = glob.glob(ROOT + "*_demos_*.pkl")

ptu.set_gpu_mode(True, 0)
os.environ['gpu_id'] = str(0)
model = torch.load(local_path).to(ptu.device)

def reconstruct_image(img):
    latent_encoding = model.encode_one_np(img).reshape(1, 720)
    image_reconstruction = model.decode_one_np(latent_encoding).reshape(6912,)
    return image_reconstruction

for file in all_files:
    print(file)
    data = np.load(file, allow_pickle=True)
    for i in range(len(data)):
        if i % 100 == 0:
            print('{} done'.format(i))
        for j in range(len(data[i]['observations'])):
            data[i]['observations'][j]['image_observation'] = reconstruct_image(data[i]['observations'][j]['image_observation'])
        for j in range(len(data[i]['next_observations'])):
            data[i]['next_observations'][j]['image_observation'] = reconstruct_image(data[i]['next_observations'][j]['image_observation'])
    
    curr_name = NEW_ROOT + 'reconstructed_' + file.split('/')[-1]
    new_file = open(curr_name, 'wb')
    pkl.dump(data, new_file)
    new_file.close()