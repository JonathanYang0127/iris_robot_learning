from widowx_envs.utils.reward_labeling_utils import get_open_index, obj_in_container_classifier
import numpy as np

def get_reward_vector(success, open_gripper_ts, max_path_len):
    if open_gripper_ts is None or not success:
        reward_list = [0] * max_path_len
    else:
        reward_list = [0] * open_gripper_ts + [1] * (max_path_len - open_gripper_ts)
    reward_vector = np.reshape(reward_list, (max_path_len, 1))
    return reward_vector

def relabel_path_rewards(object_detector, path, max_path_len):
    open_gripper_ts = get_open_index(path)
    # image = path['observations'][open_gripper_ts]['full_image']
    centroids = object_detector.go_neutral_and_get_all_centers()
    print("centroids", centroids)
    obj_in_container = obj_in_container_classifier(np.squeeze(centroids[0]))
    path['rewards'] = get_reward_vector(obj_in_container, open_gripper_ts, max_path_len)
    return path
