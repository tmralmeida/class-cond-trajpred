import csv
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

from trajectory_predictor import TrajectoryPredictor
import utils


def run_experiment(cliff_map_file, general_cliff_map_file, human_traj_file, result_file, observed_tracklet_length, planning_horizon, delta_t, r_s, pure_CVM, if_rank, beta, generate_traj_num, prob_version, exp_type, role, scene_num):
    start_length = 0

    cliff_map_data = utils.read_cliff_map_data(cliff_map_file)
    general_cliff_map_data = utils.read_cliff_map_data(general_cliff_map_file)
    human_traj_data = utils.read_magni_human_traj_data(human_traj_file)

    person_id_list = utils.get_all_person_id(human_traj_data)

    header = ["person_id", "predict_horizon", "FDE_mean", "FDE_std", "FDE_min", "FDE_1", "FDE_median", "FDE_3", "ADE_mean", "ADE_std", "ADE_min", "ADE_1", "ADE_median", "ADE_3", "top_k_ADE_by_timestep", "num_predicted_trajs"]
    with open(result_file, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

    for person_id in person_id_list:
        human_traj_data_by_person_id = utils.get_human_traj_data_by_person_id(human_traj_data, person_id)
        trajectory_predictor = TrajectoryPredictor(
            cliff_map_origin_data=cliff_map_data,
            general_cliff_map_data=general_cliff_map_data,
            human_traj_origin_data=human_traj_data_by_person_id,
            person_id=person_id,
            start_length=start_length,
            observed_tracklet_length=observed_tracklet_length,
            max_planning_horizon=planning_horizon,
            delta_t=delta_t,
            result_file=result_file,
            r_s=r_s,
            beta=beta,
            generate_traj_num=generate_traj_num,
            exp_type=exp_type,
            dataset="magni"
        )
        
        if not trajectory_predictor.check_human_traj_data(if_check=False):
            continue

        if pure_CVM:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_pure_cvm()
        else:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_stop(prob_version)

        ############ save one person's predicted trajectory ########################################
        # traj_save_dir = f"mod_based_predictors/results/magni/{exp_type}_sce{scene_num}_{role}_{person_id}"
        # utils.save_one_person_predicted_traj(all_predicted_trajectory_list, traj_save_dir)
        ############################################################################################
        
        if if_rank == 1:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_v3_rank(all_predicted_trajectory_list, version=2)
        elif if_rank == 0:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_v2(all_predicted_trajectory_list)
        elif if_rank == 2:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_most_likely(all_predicted_trajectory_list)


def main(scene_num, role, r_s, generate_traj_num, version, if_rank, beta, prob_version, exp_type):
    for test_per in [10]:
        for batch_num in tqdm(range(1,11)):
            general_cliff_map_file = f"mod_based_predictors/cliff/role_condition_k_fold_magni/general_cluster_4_grid_02/k_fold_{test_per}percentage_test/batch{batch_num}/sce{scene_num}_cliff.csv"
            if exp_type == "condition":
                cliff_map_file = f"mod_based_predictors/cliff/role_condition_k_fold_magni/role_cluster_4_grid_02/k_fold_{test_per}percentage_test/batch{batch_num}/sce{scene_num}_{role}_cliff.csv"            
            elif exp_type == "general":
                cliff_map_file = general_cliff_map_file

            magni_folder = f"data/mod_based_predictors_data/thor_magni/k_fold_{test_per}percentage_test/batch{batch_num}/test"
            human_traj_file = magni_folder + f"/sce{scene_num}_{role}.csv"
            
            observed_tracklet_length = 7
            planning_horizon = 4.8
            pure_CVM = False
            delta_t = 0.4
            if_rank = if_rank
            
            result_folder = f"mod_based_predictors/results/magni/{version}/k_fold_{test_per}percentage_test/batch{batch_num}"
            os.makedirs(result_folder, exist_ok=True)
            
            result_file = f"{result_folder}/sce{scene_num}_{role}.csv"

            run_experiment(cliff_map_file, general_cliff_map_file, human_traj_file, result_file, observed_tracklet_length, planning_horizon, delta_t, r_s, pure_CVM, if_rank, beta, generate_traj_num, prob_version, exp_type, role, scene_num)

if __name__ == "__main__":
    ############# Update param(s) from args #############
    if len(sys.argv) > 1:
        r_s = float(sys.argv[1])
        generate_traj_num = int(sys.argv[2])
        version = sys.argv[3]
        if_rank = int(sys.argv[4])
        exp_type = sys.argv[5]
        beta = int(sys.argv[6])
        prob_version = 3
        print("We update params from args: ")
        print("The version is: ", version)
        print("The r_s is: ", r_s)
        print("The generate_traj_num is: ", generate_traj_num)
        print("The version is: ", version)
        print("The if_rank is: ", if_rank)
        print("The exp_type is: ", exp_type)
    else:
        r_s = 0.2
        generate_traj_num = 3 # the k value, like in top k
        version = "cvm_origin_vel"
        if_rank = False
        prob_version = 2
        exp_type = "general"

    for scene_num in [1,2,3]:
        for role in ["carrier_box", "carrier_bucket", "carrier_lo", "visitors_alone", "visitors_g"]:
            print("scene and role: sce", scene_num, " ", role)
            main(scene_num, role, r_s, generate_traj_num, version, if_rank, beta, prob_version, exp_type)