import math
import csv
import warnings
from typing import Any, Optional

import numpy as np
import scipy

import utils
import evaluation

class TrajectoryPredictor:
    def __init__(
            self,
            *,
            cliff_map_origin_data,
            general_cliff_map_data: Optional[Any] = None,
            human_traj_origin_data,
            person_id: int = None,
            start_length: int = 0,
            observed_tracklet_length: int = 1,
            max_planning_horizon: int = 50,
            delta_t: int = 1,
            result_file: str,
            exp_num: int = 1,
            r_s : float = 1.0, # sampling radius, use when select component of SWGMM
            beta : float = 1.0, # 
            generate_traj_num : int = 10, # k value, like in top k
            exp_type : str = "general",
            dataset : str = "magni"
    ):
        self.cliff_map = cliff_map_origin_data
        self.general_cliff_map = general_cliff_map_data
        self.human_traj_data = human_traj_origin_data
        self.person_id = person_id
        self.start_length = start_length
        self.observed_tracklet_length = observed_tracklet_length
        self.max_planning_horizon = max_planning_horizon
        self.planning_horizon = None
        self.delta_t = delta_t
        self.result_file = result_file
        self.skipped_person_ids = []
        self.exp_num = exp_num
        self.r_s = r_s
        self.beta = beta
        self.generate_traj_num = generate_traj_num
        self.exp_type = exp_type
        self.dataset = dataset

    def set_planning_horizon(self):
        ground_truth_time = round(self.human_traj_data[-1,0] - self.human_traj_data[self.start_length + self.observed_tracklet_length,0], 1)
        self.planning_horizon = min(ground_truth_time, self.max_planning_horizon)

    def check_human_traj_data(self, if_check=True):
        if not if_check:
            self.planning_horizon = 4.8
            return True
        
        row_num = self.human_traj_data.shape[0]
        if row_num <= self.start_length + self.observed_tracklet_length + 1:
            return False
        if (self.human_traj_data[-1,0] - self.human_traj_data[self.start_length + self.observed_tracklet_length,0]) < self.delta_t:
            return False

        self.set_planning_horizon()
        return True

    def predict_one_human_traj_stop(self, prob_version=2):
        total_predicted_motion_list = []
        
        current_motion_origin = np.copy(self.human_traj_data[self.start_length + self.observed_tracklet_length, :])

        for _ in range(self.generate_traj_num):
            current_motion = current_motion_origin
            predicted_motion_list = np.copy(self.human_traj_data[self.start_length:self.start_length + self.observed_tracklet_length, :])
            predicted_motion_list = np.concatenate((predicted_motion_list, np.ones((predicted_motion_list.shape[0], 1))), axis=1)

            log_likelihood_sum = 0
            for time_index in range(1, int(round(self.planning_horizon / self.delta_t) + 2)):
                if self.exp_type == "general":
                    SWGMM_in_cliffmap = self._find_nearest_SWGMM_in_cliffmap(current_motion)
                elif self.exp_type == "condition":
                    SWGMM_in_cliffmap = self._find_nearest_SWGMM_in_cliffmap(current_motion, if_reuse_general_cliff_map=True)
                if SWGMM_in_cliffmap is None:
                    log_likelihood_sum += math.log(1e-5)
                    updated_motion = np.concatenate((current_motion[:5], [log_likelihood_sum]))       
                else:
                    _, sampled_velocity, new_prob = self._sample_motion_from_SWGMM_in_cliffmap(SWGMM_in_cliffmap, prob_version, current_motion)
                    updated_motion = self._apply_sampled_motion_to_current_motion(
                        sampled_velocity, current_motion, time_index
                    )
                    log_likelihood_sum += math.log(new_prob)
                    updated_motion = np.append(updated_motion, [log_likelihood_sum])
                predicted_motion_list = np.append(predicted_motion_list, [updated_motion], axis=0)
                current_motion = self._predict_with_constant_velocity_model(updated_motion)

            total_predicted_motion_list.append(predicted_motion_list)

        return total_predicted_motion_list


    def predict_one_human_traj_pure_cvm(self):
        total_predicted_motion_list = []
        current_motion_origin = np.copy(self.human_traj_data[self.start_length + self.observed_tracklet_length, :])

        for _ in range(1):
            current_motion = current_motion_origin
            predicted_motion_list = np.copy(self.human_traj_data[self.start_length:self.start_length + self.observed_tracklet_length, :])
            for time_index in range(1, int(round(self.planning_horizon / self.delta_t) + 2)):
                updated_motion = current_motion
                predicted_motion_list = np.append(predicted_motion_list, [updated_motion], axis=0)
                current_motion = self._predict_with_constant_velocity_model(updated_motion)

            total_predicted_motion_list.append(predicted_motion_list)

        return total_predicted_motion_list

    def evaluate_ADE_FDE_result_v2(self, all_predicted_trajectory_list):
        res_FDE = 0
        res_ADE = 0
        human_traj_data = self.human_traj_data[self.start_length:, :]
        start_predict_position = self.observed_tracklet_length
        error_matrix = []
        for predicted_traj in all_predicted_trajectory_list:
            error_list = evaluation.get_error_list_with_predict_timestamp(human_traj_data, predicted_traj, start_predict_position)
            error_matrix.append([round(num, 3) for num in error_list])

        max_planning_horizon = max([len(row) for row in error_matrix])

        for time_index in range(1, max_planning_horizon + 1):
            traj_error_matrix_for_one_time_index = []
            for error_list in error_matrix:
                if len(error_list) >= time_index:
                    traj_error_matrix_for_one_time_index.append(error_list[:time_index])
            num_predicted_trajs = len(traj_error_matrix_for_one_time_index)
            traj_error_array_for_one_time_index = np.array(traj_error_matrix_for_one_time_index)
            FDE_mean = round(np.mean(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_std = round(np.std(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_min = round(np.min(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_1 = round(np.quantile(traj_error_array_for_one_time_index[:,-1], 0.25), 3)
            FDE_median = round(np.quantile(traj_error_array_for_one_time_index[:,-1], 0.5), 3)
            FDE_3 = round(np.quantile(traj_error_array_for_one_time_index[:,-1], 0.75), 3)
            ADE_list = np.mean(traj_error_array_for_one_time_index, axis=1)
            ADE_mean = round(np.mean(ADE_list), 3)
            ADE_std = round(np.std(ADE_list), 3)
            ADE_min = round(np.min(ADE_list), 3)
            ADE_1 = round(np.quantile(ADE_list, 0.25), 3)
            ADE_median = round(np.quantile(ADE_list, 0.5), 3)
            ADE_3 = round(np.quantile(ADE_list, 0.75), 3)
            top_k_ADE_by_timestep = round(np.mean(np.min(traj_error_array_for_one_time_index, axis=0)), 3)
            data_row = [self.person_id, round(time_index*self.delta_t, 1), FDE_mean, FDE_std, FDE_min, FDE_1, FDE_median, FDE_3, ADE_mean, ADE_std, ADE_min, ADE_1, ADE_median, ADE_3, top_k_ADE_by_timestep, num_predicted_trajs]

            res_FDE = FDE_mean
            res_ADE = ADE_mean
                
            with open(self.result_file, "a", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_row)

        max_planning_horizon = round(max_planning_horizon * self.delta_t, 1)
        return max_planning_horizon, res_FDE, res_ADE

    def evaluate_ADE_FDE_result_v3_rank(self, all_predicted_trajectory_list, version=2):
        res_FDE = 0
        res_ADE = 0
        human_traj_data = self.human_traj_data[self.start_length:, :]
        
        start_predict_position = self.observed_tracklet_length
        error_matrix = []
        weight_matrix = []
        for predicted_traj in all_predicted_trajectory_list:
            error_list = evaluation.get_error_list_with_predict_timestamp(human_traj_data, predicted_traj, start_predict_position)
            error_matrix.append(error_list)
            weight_list = list(predicted_traj[start_predict_position+1:,-1])
            weight_matrix.append(weight_list)

        max_planning_horizon = max([len(row) for row in error_matrix])

        for time_index in range(1, max_planning_horizon + 1):
            traj_error_matrix_for_one_time_index = []
            weight_matrix_for_one_time_index = []
            for i in range(len(error_matrix)):
                error_list = error_matrix[i]
                weight_list = weight_matrix[i]
                if len(error_list) >= time_index:
                    traj_error_matrix_for_one_time_index.append(error_list[:time_index])
                    weight_matrix_for_one_time_index.append(weight_list[:time_index])

            num_predicted_trajs = len(traj_error_matrix_for_one_time_index)
            traj_error_array_for_one_time_index = np.array(traj_error_matrix_for_one_time_index)
            weight_matrix_for_one_time_index = np.array(weight_matrix_for_one_time_index)
            
            if len(weight_matrix_for_one_time_index) == 1:
                weight_matrix_for_one_time_index = np.ones((1, len(weight_matrix_for_one_time_index[0])))
            else:
                if version == 1:
                    weight_matrix_for_one_time_index = np.exp(weight_matrix_for_one_time_index)
                elif version == 2:
                    min_weight = np.min(weight_matrix_for_one_time_index, axis=0)
                    column_condition = (min_weight < 0)
                    weight_matrix_for_one_time_index[:,column_condition] = weight_matrix_for_one_time_index[:,column_condition] + np.abs(min_weight[column_condition])

                weight_matrix_for_one_time_index[weight_matrix_for_one_time_index == 0] = 1e-8
                weight_matrix_for_one_time_index = weight_matrix_for_one_time_index / weight_matrix_for_one_time_index.sum(axis=0)

            last_col_weight = weight_matrix_for_one_time_index[:, -1]
            multiply_error_weight = traj_error_array_for_one_time_index * last_col_weight[:, np.newaxis]
            multiply_error_weight = multiply_error_weight.sum(axis=0)

            FDE_mean = round(multiply_error_weight[-1], 3)
            FDE_std = round(np.std(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_min = round(np.min(traj_error_array_for_one_time_index[:,-1]), 3)
            FDE_1 = round(np.quantile(traj_error_array_for_one_time_index[:,-1], 0.25), 3)
            FDE_median = round(np.quantile(traj_error_array_for_one_time_index[:,-1], 0.5), 3)
            FDE_3 = round(np.quantile(traj_error_array_for_one_time_index[:,-1], 0.75), 3)
            ADE_list = np.mean(traj_error_array_for_one_time_index, axis=1)
            ADE_mean = round(np.mean(multiply_error_weight), 3)
            ADE_std = round(np.std(ADE_list), 3)
            ADE_min = round(np.min(ADE_list), 3)
            ADE_1 = round(np.quantile(ADE_list, 0.25), 3)
            ADE_median = round(np.quantile(ADE_list, 0.5), 3)
            ADE_3 = round(np.quantile(ADE_list, 0.75), 3)
            top_k_ADE_by_timestep = round(np.mean(np.min(traj_error_array_for_one_time_index, axis=0)), 3)
            data_row = [self.person_id, round(time_index*self.delta_t, 1), FDE_mean, FDE_std, FDE_min, FDE_1, FDE_median, FDE_3, ADE_mean, ADE_std, ADE_min, ADE_1, ADE_median, ADE_3, top_k_ADE_by_timestep, num_predicted_trajs]

            res_FDE = FDE_mean
            res_ADE = ADE_mean

            with open(self.result_file, "a", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_row)
        
        max_planning_horizon = round(max_planning_horizon * self.delta_t, 1)

        return max_planning_horizon, res_FDE, res_ADE

    def evaluate_ADE_FDE_result_most_likely(self, all_predicted_trajectory_list):
        res_FDE = 0
        res_ADE = 0
        human_traj_data = self.human_traj_data[self.start_length:, :]
        
        start_predict_position = self.observed_tracklet_length
        error_matrix = []
        weight_matrix = []
        for predicted_traj in all_predicted_trajectory_list:
            error_list = evaluation.get_error_list_with_predict_timestamp(human_traj_data, predicted_traj, start_predict_position)
            error_matrix.append(error_list)
            weight_list = list(predicted_traj[start_predict_position+1:,-1])
            weight_matrix.append(weight_list)

        max_planning_horizon = max([len(row) for row in error_matrix])

        for time_index in range(1, max_planning_horizon + 1):
            traj_error_matrix_for_one_time_index = []
            weight_matrix_for_one_time_index = []
            for i in range(len(error_matrix)):
                error_list = error_matrix[i]
                weight_list = weight_matrix[i]
                if len(error_list) >= time_index:
                    traj_error_matrix_for_one_time_index.append(error_list[:time_index])
                    weight_matrix_for_one_time_index.append(weight_list[:time_index])

            num_predicted_trajs = len(traj_error_matrix_for_one_time_index)
            traj_error_array_for_one_time_index = np.array(traj_error_matrix_for_one_time_index)
            weight_matrix_for_one_time_index = np.array(weight_matrix_for_one_time_index)

            last_col_weight = weight_matrix_for_one_time_index[:, -1]
            index_of_largest = np.argmax(last_col_weight)
            most_likely_traj_error = traj_error_array_for_one_time_index[index_of_largest]

            FDE_mean = round(most_likely_traj_error[-1], 3)
            FDE_std = 0
            FDE_min = 0
            FDE_1 = 0
            FDE_median = 0
            FDE_3 = 0
            ADE_mean = round(np.mean(most_likely_traj_error), 3)
            ADE_std = 0
            ADE_min = 0
            ADE_1 = 0
            ADE_median = 0
            ADE_3 = 0
            top_k_ADE_by_timestep = 0
            data_row = [self.person_id, round(time_index*self.delta_t, 1), FDE_mean, FDE_std, FDE_min, FDE_1, FDE_median, FDE_3, ADE_mean, ADE_std, ADE_min, ADE_1, ADE_median, ADE_3, top_k_ADE_by_timestep, num_predicted_trajs]

            res_FDE = FDE_mean
            res_ADE = ADE_mean

            with open(self.result_file, "a", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(data_row)
        
        max_planning_horizon = round(max_planning_horizon * self.delta_t, 1)

        return max_planning_horizon, res_FDE, res_ADE

    def _calculate_current_motion(self):        
        current_motion_origin = self.human_traj_data[self.start_length + self.observed_tracklet_length, :]
        sigma = 1.5
        current_speed = 0
        current_orientation = 0
        g_t = [1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(- t ** 2 / (2 * sigma ** 2)) for t in range(1, self.observed_tracklet_length + 1)]
        g_t = [g/sum(g_t) for g in g_t]
        g_t = np.flip(g_t)
        raw_speed_list = self.human_traj_data[self.start_length : self.start_length + self.observed_tracklet_length, 3]
        raw_orientation_list = self.human_traj_data[self.start_length : self.start_length + self.observed_tracklet_length, 4]
        weighted_speed_list = raw_speed_list * g_t
        current_speed = np.sum(weighted_speed_list)
        wrapped_orientation = utils.wrapTo2pi(raw_orientation_list)
        current_orientation = utils.circmean(wrapped_orientation, g_t)
        current_motion = np.concatenate((current_motion_origin[0:3], [current_speed, current_orientation]))
        return current_motion

    def _predict_with_constant_velocity_model(self, updated_motion):
        new_position = self._get_next_position_by_velocity(updated_motion[1:3], updated_motion[3:5])
        new_timestamp = np.array([round(updated_motion[0] + self.delta_t, 2)])
        predicted_motion = np.concatenate((new_timestamp, new_position, updated_motion[3:5]))
        return predicted_motion

    def _sample_motion_from_SWGMM_in_cliffmap(self, SWGMM_in_cliffmap, version, current_motion):
        if version == 3:
            SWND = self._sample_component_from_SWGMM_by_most_similer_direction(SWGMM_in_cliffmap, current_motion)
        else:
            SWND, component_weight_normalize = self._sample_component_from_SWGMM(SWGMM_in_cliffmap)

        mean = SWND[2:4]
        cov = [SWND[4:6], SWND[6:8]]

        sampled_velocity = None

        if not sampled_velocity:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                warnings.warn("runtime", RuntimeWarning)

                sampled_velocity = np.random.multivariate_normal(mean, cov, 1)

        if version == 1 or version == 3:
            try:
                prob_mvn = scipy.stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                prob = prob_mvn.pdf([sampled_velocity[0][0], sampled_velocity[0][1]])
            except ValueError:
                prob = 1 
        
        elif version == 2:
            prob = 0
            for each_SWND_index in range(len(SWGMM_in_cliffmap)):
                each_SWND = SWGMM_in_cliffmap[each_SWND_index]
                each_SWND_weight = component_weight_normalize[each_SWND_index]
                each_mean = each_SWND[2:4]
                each_cov = [each_SWND[4:6], each_SWND[6:8]]
                try:
                    each_prob_mvn = scipy.stats.multivariate_normal(mean=each_mean, cov=each_cov, allow_singular=True)
                    each_prob2 = each_prob_mvn.pdf([sampled_velocity[0][0], sampled_velocity[0][1]])
                except ValueError:
                    each_prob2 = 1
                prob += each_SWND_weight * each_prob2
            

        if prob <= 0.0:
            prob = 1e-5

        samapled_rho = sampled_velocity[0][1]
        sampled_theta = utils.wrapTo2pi(sampled_velocity[0][0])
        sampled_velocity_rho_theta = np.array([samapled_rho, sampled_theta])

        return SWND, sampled_velocity_rho_theta, prob

    def _sample_component_from_SWGMM(self, SWGMM_in_cliffmap):
        component_weight_list = []
        for SWND in SWGMM_in_cliffmap:
            component_weight_list.append(SWND[8])

        component_weight_array = np.array(component_weight_list)
        component_weight_normalize = list(component_weight_array / component_weight_array.sum())

        component_weight_acc = [np.sum(component_weight_normalize[:i]) for i in range(1, len(component_weight_normalize)+1)]
        r = np.random.uniform(0, 1)
        index = 0
        for i, threshold in enumerate(component_weight_acc):
            if r < threshold:
                index = i
                break
        
        SWND = SWGMM_in_cliffmap[index]

        return SWND, component_weight_normalize

    def map_theta(theta):
        if theta < 0:
            return theta + 2 * math.pi
        else:
            return theta

    def _sample_component_from_SWGMM_by_most_similer_direction(self, SWGMM_in_cliffmap, current_motion):
        component_prob_list = []
        for i in range(len(SWGMM_in_cliffmap)):
            theta_twopi = utils.minuspiTo2pi(current_motion[4])
            current_velocity = np.array([theta_twopi, current_motion[3]])
            mahalanobis_distance = utils.get_mahalanobis_distance(current_velocity, SWGMM_in_cliffmap[i])
            component_prob_list.append(mahalanobis_distance)

        component_prob_array = np.array(component_prob_list)
        index = np.argmin(component_prob_array)
        
        SWND = SWGMM_in_cliffmap[index]

        return SWND

    def _apply_sampled_motion_to_current_motion(self, sampled_velocity, current_motion, time_index, if_only_use_sample_velocity=False):
        current_velocity = current_motion[3:5]
        result_rho = current_velocity[0]
        
        sampled_orientation = utils.wrapTo2pi(sampled_velocity[1])
        current_orientation = utils.wrapTo2pi(current_velocity[1])
        
        delta_theta = utils.circdiff(sampled_orientation, current_orientation)
        delta_rho = np.abs(sampled_velocity[0] - current_velocity[0])

        if if_only_use_sample_velocity:
            param_lambda = 1
        else:
            param_lambda = self._apply_gaussian_kernel(delta_theta, self.beta)

        result_theta = utils.circmean([sampled_orientation, current_orientation], [param_lambda, 1-param_lambda])

        ########### If use speed from CLiFF-map ###########
        speed_beta = self.beta
        param_lambda_rho = self._apply_gaussian_kernel(delta_rho, speed_beta)
        sampled_rho = sampled_velocity[0]
        current_rho = current_velocity[0]
        result_rho = (sampled_rho - current_rho) * param_lambda_rho + current_rho
        ###################################################
        
        predicted_motion = np.concatenate(
            (current_motion[0:3], [result_rho, result_theta])
        )

        return predicted_motion

    def _apply_gaussian_kernel(self, x, beta):
        return np.exp(-beta*x**2)

    def _find_nearest_SWGMM_in_cliffmap(self, current_motion, if_reuse_general_cliff_map=False):
        cliff_map = self.cliff_map
        near_region = self.r_s
        nearest_SWGMM = []
        current_location = current_motion[1:3]
        location_array = cliff_map[:,0:2]
        nearest_index = np.argmin(np.sum(np.power(location_array - current_location, 2), axis=1))
        nearest_distance = utils.get_euclidean_distance(location_array[nearest_index], current_location)
        if nearest_distance < near_region:
            index_list = np.where((location_array[:, 0] == location_array[nearest_index][0]) & (location_array[:, 1] == location_array[nearest_index][1]))
            for index in index_list[0]:
                nearest_SWGMM.append(cliff_map[index].tolist())
        else:
            if if_reuse_general_cliff_map:
                cliff_map = self.general_cliff_map
                near_region = self.r_s
                nearest_SWGMM = []
                current_location = current_motion[1:3]
                location_array = cliff_map[:,0:2]
                nearest_index = np.argmin(np.sum(np.power(location_array - current_location, 2), axis=1))
                nearest_distance = utils.get_euclidean_distance(location_array[nearest_index], current_location)
                if nearest_distance > near_region:
                    return None
                index_list = np.where((location_array[:, 0] == location_array[nearest_index][0]) & (location_array[:, 1] == location_array[nearest_index][1]))
                for index in index_list[0]:
                    nearest_SWGMM.append(cliff_map[index].tolist())
            else:
                return None
            
        return nearest_SWGMM

    def _get_next_position_by_velocity(self, current_position, current_velocity):
        if self.dataset == "magni":
            delta_t = self.delta_t
        elif self.dataset == "sdd":
            delta_t = 1 
        
        next_position_x = current_position[0] + current_velocity[0] * np.cos(current_velocity[1]) * delta_t
        next_position_y = current_position[1] + current_velocity[0] * np.sin(current_velocity[1]) * delta_t

        return np.array([next_position_x, next_position_y])
