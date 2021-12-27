import os

import torch

from lib.vae import vae_datasets


def load_datasets(data_dir,
                  encoding_dir=None,
                  dataset_ctor=None,
                  # transform=None,
                  # crop=[48, 48],
                  # delta_t=1,
                  **kwargs,
                  ):
    if dataset_ctor is None:
        dataset_ctor = vae_datasets.VaeDataset

    train_data_path = os.path.join(data_dir, 'train_data.npy')
    test_data_path = os.path.join(data_dir, 'test_data.npy')

    if encoding_dir is None:
        train_encoding_path = None
        test_encoding_path = None
    else:
        train_encoding_path = os.path.join(encoding_dir, 'train_encoding.npy')
        test_encoding_path = os.path.join(encoding_dir, 'test_encoding.npy')

    train_dataset = dataset_ctor(
        train_data_path,
        train_encoding_path,
        # transform=None,
        # crop=[48, 48],
        **kwargs,
    )
    test_dataset = dataset_ctor(
        test_data_path,
        test_encoding_path,
        # transform=None,
        # crop=[48, 48],
        **kwargs,
    )

    return train_dataset, test_dataset


def load_model(root_dir):
    vqvae_path = os.path.join(root_dir, 'vqvae.pt')
    dynamics_path = os.path.join(root_dir, 'dynamics.pt')
    classifier_path = os.path.join(root_dir, 'classifier.pt')

    vqvae = torch.load(vqvae_path)
    dynamics = torch.load(dynamics_path)

    if os.path.exists(classifier_path):
        classifier = torch.load(classifier_path)
    else:
        classifier = None

    return {
        'vqvae': vqvae,
        'dynamics': dynamics,
        'classifier': classifier,
    }
