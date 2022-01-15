import matplotlib.pyplot as plt
import argparse
import numpy as np
import itertools
from matplotlib.patches import Rectangle

def get_corners_from_pos_half_extents(pos, half_extents):
    corners = []
    half_x, half_y = half_extents
    dxdys = list(itertools.product((-half_x, half_x), (-half_y, half_y)))
    for dxdy in dxdys:
        corners.append(pos[:2] + np.array(dxdy))
    return corners

def draw_rectangle(corners, half_extents, color, label):
    low_x = min(corners, key=lambda xy: xy[0])[0]
    low_y = min(corners, key=lambda xy: xy[1])[1]
    width = 2 * half_extents[0]
    height = 2 * half_extents[1]
    plt.gca().add_patch(Rectangle((low_x, low_y), width, height,
        edgecolor=color,
        facecolor='none',
        lw=2,
        label=label))

def plot(data):
    # Get all reverse task indices. Create array with only those rows.
    reverse_rollout_idxs = np.where(data[:,1] == 1)[0]
    import ipdb; ipdb.set_trace()
    xy_pos = data[reverse_rollout_idxs, 3:5]
    x, y = xy_pos[:,0], xy_pos[:,1]

    object_position_low=[0.45, .14, -0.3],
    object_position_high=[0.73, 0.35, -0.3]

    tray_pos = (np.array(object_position_low) + np.array(object_position_high)).flatten() / 2
    tray_half_extents = (np.array(object_position_high) - tray_pos).flatten()[:2]
    tray_corners = get_corners_from_pos_half_extents(tray_pos, tray_half_extents)
    draw_rectangle(tray_corners, tray_half_extents, "red", "Tray")

    container_pos = np.array([.52, 0.25, -.30])
    container_half_extents = (0.075, 0.105)
    container_corners = get_corners_from_pos_half_extents(container_pos, container_half_extents)
    draw_rectangle(container_corners, container_half_extents, "green", "Container")

    plt.scatter(x, y, alpha=0.1, label="Init target obj pos")
    plt.gca().invert_xaxis()
    plt.title("Initial Positions of Target Object on Reverse Tasks")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.savefig("hi.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj-infos-arr", type=str, default="")
    args = parser.parse_args()

    data = np.load(args.obj_infos_arr)
    plot(data)
