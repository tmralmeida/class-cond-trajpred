import numpy as np

import utils


def get_position_in_ground_truth(timestamp, traj):
    if timestamp > traj[-1,0]:
        return None
    diff_array = np.abs(traj[:, 0] - timestamp)
    min_time_diff_index = np.argmin(diff_array)
    nearest_time = traj[min_time_diff_index][0]
    if diff_array[min_time_diff_index] == 0:
        return traj[min_time_diff_index][1:3]
    
    if timestamp > nearest_time:
        i1 = min_time_diff_index
        i2 = min_time_diff_index + 1
    if timestamp < nearest_time:
        i1 = min_time_diff_index - 1
        i2 = min_time_diff_index

    proportion = (timestamp - traj[i1][0]) / (traj[i2][0] - traj[i1][0])
    x = proportion * (traj[i2][1] - traj[i1][1]) + traj[i1][1]
    y = proportion * (traj[i2][2] - traj[i1][2]) + traj[i1][2]

    return np.array([x, y])


def get_error_list_with_predict_timestamp(ground_truth_traj, predicted_traj, start_predict_position):
    ADE = []

    for predicted_point in predicted_traj[start_predict_position+1:]:
        timestamp = round(predicted_point[0],2)
        
        predicted_position = predicted_point[1:3]
        ground_truth_position = get_position_in_ground_truth(timestamp, ground_truth_traj)

        if ground_truth_position is None:
            break
        ADE.append(round(utils.get_euclidean_distance(ground_truth_position, predicted_position), 5))

    return ADE