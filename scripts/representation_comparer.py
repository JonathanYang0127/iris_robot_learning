import os.path as osp
import numpy as np
from matplotlib import pyplot as plt
import torch
from torchvision.utils import save_image
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.vae.vq_vae import VQ_VAE, VAE, CCVAE
from rlkit.torch.gan.bigan import BiGAN, CVBiGAN

class ModelComparison:
    def __init__(self, model_paths, datapath, num_samples=1, channels=3, imsize=48, normalize=True):
        self.models = [load_local_or_remote_file(p).cpu() for p in model_paths]
        self.data = load_local_or_remote_file(datapath).item()['observations']
        
        self.num_samples = num_samples
        self.normalize = normalize
        self.channels = channels
        self.imsize = imsize
        
        self.num_traj = self.data.shape[0]
        self.horizon = self.data.shape[1]
        self.log_dir = '/home/ashvin/data/sasha/model_comparisons/'
    
    def sample_data(self):
        i = np.random.randint(self.data.shape[0] // 2, high=self.num_traj, size=self.num_samples)
        j = np.random.randint(0, high=self.horizon, size=self.num_samples)
        constant = 255. if self.normalize else 1.
        imgs = self.data[i, j, :] / constant
        x_0 = self.data[i, 0, :] / constant
        return imgs, x_0

    def format_img(self, img):
    	formatted_imgs = ptu.from_numpy(img).reshape(self.num_samples, self.channels, self.imsize, self.imsize).transpose(2, 3)
    	return formatted_imgs

    def run_model(self, model, imgs, x_0):
        if isinstance(model, VQ_VAE):
            recon = model.decode_np(model.encode_np(imgs))
            cond = model.encode_np(x_0)
            sample = model.decode_np(model.sample_prior(self.num_samples, cond=cond))
        elif isinstance(model, CCVAE) or isinstance(model, CVBiGAN):
            recon = model.decode_np(model.encode_np(imgs, cond=x_0))
            sample = model.decode_np(model.sample_prior(self.num_samples, cond=x_0))
        else:
            recon = model.decode_np(model.encode_np(imgs))
            sample = model.decode_np(model.sample_prior(self.num_samples))
        return recon, sample
    
    def compare_models(self, imgs, x_0):
        recons = [self.format_img(imgs)]
        samples = [self.format_img(x_0)]

        for model in self.models:
            recon, sample = self.run_model(model, imgs, x_0)
            recons.append(self.format_img(recon))
            samples.append(self.format_img(sample))
        
        return torch.cat(recons, dim=0), torch.cat(samples, dim=0)
    
    def dump(self, recons, samples):
        recon_dir = osp.join(self.log_dir, 'recons.png')
        sample_dir = osp.join(self.log_dir, 'samples.png')
        
        save_image(recons.data.cpu(), recon_dir, nrow=self.num_samples)
        save_image(samples.data.cpu(), sample_dir, nrow=self.num_samples)
    
    def forward(self):
        imgs, x_0 = self.sample_data()
        recons, samples = self.compare_models(imgs, x_0)
        self.dump(recons, samples)

if __name__ == "__main__":
    path_maker = lambda model_type: 'sasha/models/{0}/{1}.pt'.format(model_type, model_type)
    #model_paths = [path_maker('vqvae'), path_maker('ccvae'), path_maker('cbigan'), path_maker('vae'), path_maker('bigan')]
    # train_datapath = 'sasha/affordances/combined/combined_images.npy'
    # test_datapath = 'sasha/affordances/combined/combined_test_images.npy'

    test_datapath = 'sasha/complex_obj/gr_test_complex_obj_images.npy'
    model_paths = [
    "sasha/complex_obj/pixelcnn_vqvae.pkl",
    "sasha/complex_obj/baselines/ccvae.pkl",
    "sasha/models/cbigan/cbigan.pkl",
    "sasha/complex_obj/baselines/vae.pkl",
    "sasha/models/bigan/bigan.pkl"
    ]
 	
    comparer = ModelComparison(model_paths, test_datapath)
    comparer.forward()
