import skvideo.io
import skvideo.datasets
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from rlkit.torch import pytorch_util as ptu

def output_filmstrip(videodata, i, j, cols=5, rows=8, timestep=8, num_steps=4, frames=[]):
    i_adj = 6
    j_adj = 14
    
    imsize = 48
    white_width = 5
    start_i, end_i = i*imsize + i_adj, (i+1)*imsize + i_adj
    start_j, end_j = j*imsize + j_adj, (j+1)*imsize +j_adj

    video_section = videodata[:, start_i:end_i, start_j:end_j, :]
    whitespace = np.ones((end_i - start_i, white_width, 3)) * 255

    images = []
    for t in range(num_steps):
        curr_frame = timestep * t if not frames else frames[t]
        images.append(whitespace)
        images.append(video_section[curr_frame])
    images.append(whitespace)

    images = np.uint8(np.concatenate(images, axis=1))
    images = Image.fromarray(images)
    images.save('/home/ashvin/data/sasha/filmstrip/filmstrip.jpeg')


if __name__ == "__main__":
    # videodata = skvideo.io.vread("/home/ashvin/data/sasha/filmstrip/task_1.mp4")
    # output_filmstrip(videodata, 6, 0, frames=[0, 8, 16, 30])
    # videodata = skvideo.io.vread("/home/ashvin/data/sasha/filmstrip/task_2.mp4")
    # output_filmstrip(videodata, 6, 4, frames=[0, 16, 20, 30])
    # videodata = skvideo.io.vread("/home/ashvin/data/sasha/filmstrip/task_3.mp4")
    # output_filmstrip(videodata, 6, 3, frames=[0, 13, 16, 20])
    videodata = skvideo.io.vread("/home/ashvin/data/sasha/filmstrip/randobj_env.mp4")
    output_filmstrip(videodata, 2, 3, frames=[0, 2, 4, 30])