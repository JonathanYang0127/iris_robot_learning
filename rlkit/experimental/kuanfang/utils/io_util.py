import os

import torch
from torch.utils.data import DataLoader

from rlkit.torch import pytorch_util as ptu  # NOQA

from rlkit.experimental.kuanfang.vae import vae_datasets


def load_datasets(data_dir,
                  encoding_dir=None,
                  dataset_ctor=None,
                  keys=['train', 'test'],
                  vqvae_mode='zi',
                  is_val_format=False,
                  **kwargs,
                  ):
    if dataset_ctor is None:
        dataset_ctor = vae_datasets.VaeDataset

    datasets = {}
    for key in keys:
        if is_val_format:
            if key == 'train':
                data_path = os.path.join(data_dir, 'combined_images.npy')
            elif key == 'test':
                data_path = os.path.join(data_dir, 'combined_test_images.npy')
            else:
                raise ValueError
        else:
            data_path = os.path.join(data_dir, '%s_data.npy' % (key))

        if encoding_dir is None:
            encoding_path = None
        else:
            if vqvae_mode == 'zq':
                encoding_path = os.path.join(encoding_dir,
                                             '%s_encoding.npy' % (key))
            elif vqvae_mode == 'zi':
                encoding_path = os.path.join(encoding_dir,
                                             '%s_zi.npy' % (key))
            else:
                raise ValueError

        dataset = dataset_ctor(
            data_path,
            encoding_path,
            is_val_format=is_val_format,
            **kwargs,
        )

        datasets[key] = dataset

    return datasets


def data_loaders(train_data, test_data, batch_size):
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True)
    return train_loader, test_loader


def load_model(root_dir):
    vqvae_path = os.path.join(root_dir, 'vqvae.pt')
    affordance_path = os.path.join(root_dir, 'affordance.pt')
    classifier_path = os.path.join(root_dir, 'classifier.pt')
    discriminator_path = os.path.join(root_dir, 'discriminator.pt')

    vqvae = torch.load(vqvae_path).to(ptu.device)
    affordance = torch.load(affordance_path).to(ptu.device)

    if os.path.exists(classifier_path):
        classifier = torch.load(classifier_path).to(ptu.device)
    else:
        classifier = None

    if os.path.exists(discriminator_path):
        discriminator = torch.load(discriminator_path).to(ptu.device)
    else:
        discriminator = None

    return {
        'vqvae': vqvae,
        'affordance': affordance,
        'classifier': classifier,
        'discriminator': discriminator,
    }
