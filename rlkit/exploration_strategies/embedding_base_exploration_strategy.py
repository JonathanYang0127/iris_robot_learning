import torch
import rlkit.torch.pytorch_util as ptu
import numpy as np

class BaseExplorationStrategy:
    def __init__(self, embeddings, policy=None, q_function=None):
        self.embeddings = embeddings
        if len(embeddings.shape) == 3:
            t, b, _ = self.embeddings.shape
            self.embeddings_batch = self.embeddings.reshape(t * b, -1)
        else:
            self.embeddings_batch = embeddings
        self.policy=policy
        self.q_function=q_function

    def sample_embedding(self, **kwargs):
        pass

    def post_trajectory_update(self, **kwargs):
        pass

    @staticmethod
    def compute_embeddings(replay_buffer_positive, encoder, num_embeddings_per_task=200):
        task_idxs = replay_buffer_positive.task_indices
        contexts = replay_buffer_positive.sample_batch(task_idxs, num_embeddings_per_task)['observations']
        t, b, _ = contexts.shape
        contexts = ptu.from_numpy(contexts)
        embeddings = encoder.encoder_net(contexts)[1].detach().cpu().numpy()
        embeddings = embeddings.reshape(t, b, -1)

        return embeddings

    def run_network_batch(self, network, batch, batch_size=1000, post_func=lambda x: x):
        output = []
        for i in range(batch.shape[0] // batch_size):
            minibatch = batch[i * batch_size: (i + 1) * batch_size]
            minibatch = ptu.from_numpy(minibatch)
            out = network(minibatch)
            output.extend(post_func(out).detach().cpu().numpy())
            minibatch.cpu()

        if batch.shape[0] % batch_size != 0:
            final_idx = batch.shape[0] // batch_size
            minibatch = batch[final_idx * batch_size:]
            minibatch = ptu.from_numpy(minibatch)
            out = network(minibatch)
            output.extend(post_func(out).detach().cpu().numpy())
            minibatch.cpu()

        return np.array(output)
