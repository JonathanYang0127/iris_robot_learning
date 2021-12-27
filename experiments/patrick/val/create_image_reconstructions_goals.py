import torch
import numpy as np
from rlkit.torch import pytorch_util as ptu
import os
import glob
import pickle as pkl

ROOT = '/2tb/home/patrickhaoy/data/affordances/data/reset_free_v5_rotated_top_drawer/'
local_path = ROOT + 'best_vqvae.pt'
file = ROOT + 'top_drawer_goals.pkl'

ptu.set_gpu_mode(True, 0)
os.environ['gpu_id'] = str(0)
model = torch.load(local_path).to(ptu.device)

def reconstruct_image(img):
    latent_encoding = model.encode_one_np(img).reshape(1, 720)
    image_reconstruction = model.decode_one_np(latent_encoding).reshape(6912,)
    return image_reconstruction

print(file)
data = np.load(file, allow_pickle=True)
keys = ['image_desired_goal']#, 'initial_image_observation']
for k in keys:
    assert k in data.keys()

for k in keys:
    print(k)
    for i in range(data[k].shape[0]):
        if i % 100 == 0:
            print('{} done'.format(i))
        data[k][i] = reconstruct_image(data[k][i])

curr_name = ROOT + 'reconstructed_' + file.split('/')[-1]
new_file = open(curr_name, 'wb')
pkl.dump(data, new_file)
new_file.close()