import numpy as np
import matplotlib.pyplot as plt

import utils


def plot_human_traj(human_traj_data, observed_tracklet_length=8):
    plt.plot(human_traj_data[:observed_tracklet_length+1, 1], human_traj_data[:observed_tracklet_length+1, 2], color="limegreen", label="Observation", lw=3)
    plt.scatter(human_traj_data[:observed_tracklet_length+1, 1], human_traj_data[:observed_tracklet_length+1, 2], marker='o', alpha=1, color="limegreen", s=5)

    plt.plot(human_traj_data[observed_tracklet_length:, 1], human_traj_data[observed_tracklet_length:, 2], color="r", label="Ground truth", lw=3)
    plt.scatter(human_traj_data[observed_tracklet_length:, 1], human_traj_data[observed_tracklet_length:, 2], marker='o', alpha=1, color="r", s=5)


def plot_most_likely_trajectory(total_predicted_motion_list, observed_tracklet_length=8):
    weight_list = []
    for predicted_traj in total_predicted_motion_list:
        weight_list.append(predicted_traj[-1, -1])
    index_of_largest = weight_list.index(max(weight_list))
    
    predicted_traj = total_predicted_motion_list[index_of_largest]

    plt.plot(predicted_traj[:, 1], predicted_traj[:, 2], 'b', alpha=1)
    for i in range(0, observed_tracklet_length):
        plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=10)
    plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color="b", marker="o", s=10)


def plot_most_likely_traj_blue_colors(total_predicted_motion_list, observed_tracklet_length=8):
    filtered_predicted_trajs = []
    weight_list = []
    for predicted_traj in total_predicted_motion_list:
        filtered_predicted_trajs.append(predicted_traj)
        weight_list.append(predicted_traj[-1, -1])
    weight_list = np.array(weight_list)
    sorted_weight_list = np.sort(weight_list)[::-1]

    colors = ["navy", "mediumblue", "blue", "royalblue", "cornflowerblue", "deepskyblue", "skyblue", "lightskyblue", "powderblue"]
    for j in range(len(filtered_predicted_trajs)):
        predicted_traj = filtered_predicted_trajs[j]
        for i in range(0, observed_tracklet_length):
            plt.scatter(predicted_traj[i, 1], predicted_traj[i, 2], color="limegreen", marker="o", s=10)
            
        color_index = np.where(sorted_weight_list==weight_list[j])[0][0]
        if color_index >= len(colors):
            color_index = len(colors) - 1
        plt.plot(predicted_traj[:, 1], predicted_traj[:, 2], color=colors[color_index], lw=3)
        plt.scatter(predicted_traj[observed_tracklet_length:, 1], predicted_traj[observed_tracklet_length:, 2], color=colors[color_index], marker="o", s=30)


def plot_cliff_map(cliff_map_data):
    (u, v) = utils.pol2cart(cliff_map_data[:, 3], cliff_map_data[:, 2])
    color = cliff_map_data[:, 2]
    plt.quiver(cliff_map_data[:, 0], cliff_map_data[:, 1], u, v, color, alpha=1, cmap="hsv")
