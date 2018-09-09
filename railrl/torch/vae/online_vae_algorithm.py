from railrl.core import logger
from railrl.data_management.shared_obs_dict_replay_buffer \
        import SharedObsDictRelabelingBuffer
import railrl.torch.vae.vae_schedules as vae_schedules
from railrl.torch.torch_rl_algorithm import TorchRLAlgorithm
from railrl.torch.vae.conv_vae import ConvVAE
import railrl.torch.pytorch_util as ptu

from torch.multiprocessing import Process, Pipe
from multiprocessing.connection import wait

from threading import Thread
from time import sleep

class OnlineVaeAlgorithm(TorchRLAlgorithm):

    def __init__(
        self,
        vae,
        vae_trainer,
        vae_save_period=1,
        vae_training_schedule=vae_schedules.never_train,
        oracle_data=False,
        parallel_vae_train=True,
    ):
        self.vae = vae
        self.vae_trainer = vae_trainer
        self.vae_trainer.model = self.vae
        self.vae_save_period = vae_save_period
        self.vae_training_schedule = vae_training_schedule
        self.epoch = 0
        self.oracle_data = oracle_data

        self.vae_training_process = None
        self.update_vae_thread = None
        self.parallel_vae_train = parallel_vae_train

    def _post_epoch(self, epoch):
        super()._post_epoch(epoch)
        if self.parallel_vae_train and self.vae_training_process is None:
            self.init_vae_training_subproces()

        should_train, amount_to_train = self.vae_training_schedule(epoch)
        if should_train:
            if self.parallel_vae_train:
                assert self.vae_training_process.is_alive()
                # Make sure the last vae update has finished before starting
                # another one
                if self.update_vae_thread is not None:
                    self.update_vae_thread.join()
                self.update_vae_thread = Thread(
                    target=OnlineVaeAlgorithm.update_vae_thread,
                    args=(self,)
                )
                self.update_vae_thread.start()
                self.vae_conn_pipe.send((amount_to_train, epoch))
            else:
                self.vae.train()
                _train_vae(
                    self.vae_trainer,
                    self.replay_buffer,
                    epoch,
                    amount_to_train
                )
                self.vae.eval()
                self.replay_buffer.refresh_latents(epoch)
                _test_vae(
                    self.vae_trainer,
                    self.epoch,
                    vae_save_period=self.vae_save_period
                )
        # very hacky
        self.epoch = epoch + 1

    def _post_step(self, step):
        pass

    def reset_vae(self):
        self.vae.init_weights(self.vae.init_w)

    @property
    def networks(self):
        return [self.vae]

    def update_epoch_snapshot(self, snapshot):
        snapshot.update(vae=self.vae)

    def cleanup(self):
        self.vae_training_process.terminate()

    def init_vae_training_subproces(self):
        assert isinstance(self.replay_buffer, SharedObsDictRelabelingBuffer)

        self.vae_conn_pipe, process_pipe = Pipe()
        self.vae_training_process = Process(
            target=subprocess_vae_loop,
            args=(
                process_pipe,
                self.vae,
                self.vae.state_dict(),
                self.replay_buffer,
                dict(
                    shared_obs_info=self.replay_buffer._shared_obs_info,
                    shared_next_obs_info=self.replay_buffer._shared_next_obs_info,
                    shared_size=self.replay_buffer._shared_size,
                ),
                ptu.gpu_enabled(),
            )
        )
        self.vae_training_process.start()
        self.vae_conn_pipe.send(self.vae_trainer)

    def update_vae_thread(self):
        self.vae.load_state_dict(self.vae_conn_pipe.recv())
        _test_vae(
            self.vae_trainer,
            self.epoch,
            vae_save_period=self.vae_save_period
        )

def _train_vae(vae_trainer, replay_buffer, epoch, batches=50, oracle_data=False):
    batch_sampler = replay_buffer.random_vae_training_data
    if oracle_data:
        batch_sampler = None
    vae_trainer.train_epoch(
        epoch,
        sample_batch=batch_sampler,
        batches=batches,
        from_rl=True,
    )
    replay_buffer.train_dynamics_model(batches=batches)

def _test_vae(vae_trainer, epoch, vae_save_period=1):
    save_imgs = epoch % vae_save_period == 0
    vae_trainer.test_epoch(
        epoch,
        from_rl=True,
        save_reconstruction=save_imgs,
    )
    if save_imgs: vae_trainer.dump_samples(epoch)

def subprocess_vae_loop(
    conn_pipe,
    vae,
    vae_params,
    replay_buffer,
    shared_vars,
    use_gpu=True,
):
    """
    The observations and next_observations of the replay buffer are stored in
    shared memory. This loop waits until the parent signals to start vae
    training, trains and sends the vae back, and then refreshes the latents.
    Refreshing latents in the subprocess reflects in the main process as well
    since the latents are in shared memory. Since this is does asynchronously,
    it is possible for the main process to see half the latents updated and half
    not.
    """
    ptu.set_gpu_mode(use_gpu)
    vae_trainer = conn_pipe.recv()
    vae.load_state_dict(vae_params)
    if ptu.gpu_enabled():
        vae.cuda()
    vae_trainer.set_vae(vae)
    replay_buffer.init_from(
        shared_vars['shared_obs_info'],
        shared_vars['shared_next_obs_info'],
        shared_vars['shared_size']
    )
    replay_buffer.env.vae = vae
    while True:
        amount_to_train, epoch = conn_pipe.recv()
        vae.train()
        _train_vae(vae_trainer, replay_buffer, epoch, amount_to_train)
        vae.eval()
        conn_pipe.send(vae_trainer.model.state_dict())
        replay_buffer.refresh_latents(epoch)
