"""
"""
import argparse

import joblib


def main(dataset_path, pause=False):
    """
    :param dataset_path: Path to serialized data.
    :return:
    """
    data = joblib.load(dataset_path)
    import ipdb; ipdb.set_trace()
    replay_buffer = data['replay_buffer']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('replay_pkl_path', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--pause', action='store_true')
    args = parser.parse_args()

    dataset_path = args.replay_pkl_path
    main(dataset_path, args.pause)
