from rlkit.misc.asset_loader import load_local_or_remote_file
from rlkit.torch.vae.vq_vae import VQ_VAE, VAE, CCVAE
import numpy as np
import torch

class RepresentationComparer:
	def __init__(self, model_paths, datapath):
		self.models = [load_local_or_remote_file(p) for p in model_paths]
		self.data = load_local_or_remote_file(datapath).item()
		self.num_traj = datapath['observations'].shape[0]
		self.horizon = datapath['observations'].shape[1]


	def sample_imgs(self, n=8):
		i = np.random.randint(0, self.num_traj, n)
		j = np.random.randint(0, self.num_traj, n)
		imgs = self.datapath['observations'][i, j, :]
		x_0 = self.datapath['observations'][i, 0, :]
		return imgs, x_0

	# def reconstruct(self, z)

	def forward(self, imgs, x_0):
		recons, samples = [imgs], [x_0]
		for model in self.models:
			if isinstance(model, CCVAE):
				z = model.encode_np(imgs, x_0)
			else:
				z = model.encode_np(imgs)

			if isinstance(model, VQ_VAE):
				z_0 = model.encode_np(x_0)
				z_s = model.sample_prior(1, cond=z_0)
			elif isinstance(model, CCVAE):
				z_s = model.sample_prior(1, cond=x_0)
			elif isinstance(model, VAE):
				z_s = model.sample_prior(1, cond=x_0)

			except:
				z = model.encode_np(imgs, x_0)
			recons.append(model.decode_np(z))
			try:
				z_s = model.sample_prior(1)
			except:
				z_s = model.sample_prior(1, )
			samples.append()
