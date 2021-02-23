import numpy as np
import matplotlib.pyplot as plt
import time
import glob
from rlkit.misc.asset_loader import load_local_or_remote_file

def crop(img):
    return img[::10, 50:530:10, :].transpose([2, 1, 0]).flatten()

pretrained_vae_path="/home/ashvin/data/s3doodad/ashvin/icra2021/widowx/sawyer-exp/run0/id0/itr_1500.pt"
# load_local_or_remote_file(pretrained_vae_path)

for filename in glob.glob("/home/ashvin/data/s3doodad/demos/icra2021/v1/obj*.npy"):
    print(filename)
    data = np.load(filename, allow_pickle=True)

    x = []
    x0 = []

    for traj_i in range(len(data)):
        traj = data[traj_i]["observations"]
        print(traj_i, len(traj))
        img0 = crop(traj[0]["image_observation"])
        for t in range(len(traj)):
            # print("frame", t)
            if not traj[t]:
                print(traj_i, t)
                continue

            img = crop(traj[t]["image_observation"])
            x.append(img)
            x0.append(img0)

    goals = {'image_desired_goal': x, 'initial_image_observation': x0}
    new_filename = "/home/ashvin/data/s3doodad/demos/icra2021/v1/goals.npy"
    np.save(new_filename, goals)