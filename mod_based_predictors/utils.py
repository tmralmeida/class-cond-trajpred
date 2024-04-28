import os
import numpy as np
import pandas as pd
from math import dist
import math


def get_all_person_id(data):
    person_id_list = list(data.person_id.unique())
    return person_id_list


def get_human_traj_data_by_person_id(human_traj_origin_data, person_id):
    human_traj_data_by_person_id = human_traj_origin_data.loc[human_traj_origin_data['person_id'] == person_id]
    human_traj_array = human_traj_data_by_person_id[["time", "x", "y", "velocity", "motion_angle"]].to_numpy()

    return human_traj_array


def save_one_person_predicted_traj(all_predicted_trajectory_list, traj_save_dir):
    weight_list = []
    for predict_traj in all_predicted_trajectory_list:
        weight_list.append(predict_traj[-1][-1])

    index_of_largest = weight_list.index(max(weight_list))
    ml_predict_traj = all_predicted_trajectory_list[index_of_largest]
    
    ml_predict_traj[:, 0:5] = np.around(ml_predict_traj[:, 0:5], decimals=1)

    os.makedirs(traj_save_dir, exist_ok=True)
    np.savetxt(f"{traj_save_dir}/traj.csv", ml_predict_traj, delimiter=",", fmt='%g')
    

def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def millimeter_to_meter(data, column_names):
    for column_name in column_names:
        data[column_name] = data[column_name].apply(lambda x: x / 1000)
    return data


def pixel_to_meter(data, column_names):
    for column_name in column_names:
        data[column_name] = data[column_name].apply(lambda x: x * 0.0247)
    return data


def get_euclidean_distance(position_array_1, position_array_2):
    return dist(position_array_1, position_array_2)


def get_euclidean_distance_point(x1, y1, x2, y2):
    return dist((x1, y1), (x2, y2))


def get_mahalanobis_distance(point, SWGMM):
    mean = SWGMM[2:4]
    cov = [SWGMM[4:6], SWGMM[6:8]]
    cov = np.array(cov)

    ## For circular value
    theta_diff = circdiff(point[0], mean[0])
    speed_diff = point[1] - mean[1]
    diff = np.array([theta_diff, speed_diff])

    try:
        mahalanobis_distance = np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)
    except:
        cov += np.eye(cov.shape[0]) * 1e-5
        mahalanobis_distance = np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff)

    return mahalanobis_distance


def read_role_sdd_human_traj_data(human_traj_file):
    data = pd.read_csv(human_traj_file)
    data = data[["time", "ag_id", "x", "y", "speed", "theta_delta"]]
    data.rename(columns={'ag_id': 'person_id'}, inplace=True)
    data.rename(columns={'speed': 'velocity'}, inplace=True)
    data.rename(columns={'theta_delta': 'motion_angle'}, inplace=True)

    return data


def read_magni_human_traj_data(human_traj_file):
    data = pd.read_csv(human_traj_file)
    data = data[["time", "ag_id", "x", "y", "speed", "theta_delta"]]
    data.rename(columns={'ag_id': 'person_id'}, inplace=True)
    data.rename(columns={'speed': 'velocity'}, inplace=True)
    data.rename(columns={'theta_delta': 'motion_angle'}, inplace=True)

    return data


def read_cliff_map_data(cliff_map_file):
    data = pd.read_csv(cliff_map_file, header=None)
    data.columns = ["x", "y", "motion_angle", "velocity",
                    "cov1", "cov2", "cov3", "cov4", "weight",
                    "motion_ratio", "observation_ratio"]

    return data.to_numpy()


def _circfuncs_common(samples, high, low):
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    sin_samp = np.sin((samples - low)*2.* np.pi / (high - low))
    cos_samp = np.cos((samples - low)*2.* np.pi / (high - low))

    return samples, sin_samp, cos_samp


def circmean(samples, weights, high=2*np.pi, low=0):
    samples = np.asarray(samples)
    weights = np.asarray(weights)
    samples, sin_samp, cos_samp = _circfuncs_common(samples, high, low)
    sin_sum = sum(sin_samp * weights)
    cos_sum = sum(cos_samp * weights)
    res = np.arctan2(sin_sum, cos_sum)
    res = res*(high - low)/2.0/np.pi + low
    return wrapTo2pi(res)


def circdiff(circular_1, circular_2):
    res = np.arctan2(np.sin(circular_1-circular_2), np.cos(circular_1-circular_2))
    return abs(res)


def wrapTo2pi(circular_value):
    return np.round(np.mod(circular_value,2*np.pi), 3)


def minuspiTo2pi(theta):
    if theta < 0:
        return theta + 2 * math.pi
    else:
        return theta
