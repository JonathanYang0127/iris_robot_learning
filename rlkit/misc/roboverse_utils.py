import numpy as np
import os
import os.path as osp
from rlkit.core import logger


def process_keys(observations, observation_keys):
    output = []
    for i in range(len(observations)):
        observation = dict()
        for key in observation_keys:
            if key == 'image':
                image = observations[i]['image']
                if len(image.shape) == 3:
                    image = np.transpose(image, [2, 0, 1])
                    image = (image.flatten())/255.0
                else:
                    print('image shape: {}'.format(image.shape))
                    raise ValueError
                observation[key] = image
            elif key == 'state':
                observation[key] = np.array(observations[i]['state'])
            else:
                raise NotImplementedError
        output.append(observation)
    return output


def add_data_to_buffer(data, replay_buffer, observation_keys):

    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_keys(data[j]['observations'], observation_keys),
            next_observations=process_keys(
                data[j]['next_observations'], observation_keys),
        )
        replay_buffer.add_path(path)


# TODO(avi): Fold this in with the function above
def add_data_to_buffer_multitask(data, replay_buffer, observation_keys, task):

    for j in range(len(data)):
        assert (len(data[j]['actions']) == len(data[j]['observations']) == len(
            data[j]['next_observations']))

        path = dict(
            rewards=[np.asarray([r]) for r in data[j]['rewards']],
            actions=data[j]['actions'],
            terminals=[np.asarray([t]) for t in data[j]['terminals']],
            observations=process_keys(data[j]['observations'], observation_keys),
            next_observations=process_keys(
                data[j]['next_observations'], observation_keys),
        )
        replay_buffer.add_path(task, path)


class VideoSaveFunctionBullet:
    def __init__(self, variant):
        self.logdir = logger.get_snapshot_dir()
        self.dump_video_kwargs = variant.get("dump_video_kwargs", dict())
        self.save_period = self.dump_video_kwargs.pop('save_video_period', 5)

    def __call__(self, algo, epoch):
        if epoch % self.save_period == 0 or epoch == algo.num_epochs:
            video_dir = osp.join(self.logdir,
                                 'videos_eval/{epoch}/'.format(epoch=epoch))
            eval_paths = algo.eval_data_collector.get_epoch_paths()
            dump_video_basic(video_dir, eval_paths)


def dump_video_basic(video_dir, paths):
    if not os.path.exists(video_dir):
        os.makedirs(video_dir)

    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    for i, path in enumerate(paths):
        video = path['next_observations']
        frame_list = []
        for frame in video:
            # TODO(avi) Figure out why this hack is needed
            # if isinstance(frame, np.ndarray):
            #     if 'real_image' in frame[0]:
            #         frame_list.append(frame[0]['real_image'])
            #     else:
            #         frame_list.append(frame[0]['image'])
            # else:
            #     if 'real_image' in frame:
            #         frame_list.append(frame['real_image'])
            #     else:
            #         frame_list.append(frame['image'])

            frame_list.append(frame['image'])
        frame_list = np.asarray(frame_list)
        video_len = frame_list.shape[0]
        n_channels = 3
        imsize = int(np.sqrt(frame_list.shape[1] / n_channels))
        assert imsize*imsize*n_channels == frame_list.shape[1]

        video = frame_list.reshape(video_len, n_channels, imsize, imsize)
        video = np.transpose(video, (0, 2, 3, 1))
        video = (video*255.0).astype(np.uint8)
        filename = osp.join(video_dir, '{}.mp4'.format(i))
        FPS = float(np.ceil(video_len/3.0))
        writer = cv2.VideoWriter(filename, fourcc, FPS, (imsize, imsize))
        for j in range(video.shape[0]):
            writer.write(cv2.cvtColor(video[j], cv2.COLOR_RGB2BGR))
        writer = None


def get_buffer_size(data):
    num_transitions = 0
    for i in range(len(data)):
        for j in range(len(data[i]['observations'])):
            num_transitions += 1
    return num_transitions
