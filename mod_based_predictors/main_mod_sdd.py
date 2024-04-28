import csv
import sys
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import *

from trajectory_predictor import TrajectoryPredictor
import plot_figures
import utils


def run_experiment(cliff_map_file, general_cliff_map_file, human_traj_file, result_file, observed_tracklet_length, planning_horizon, delta_t, r_s, pure_CVM, if_rank, beta, generate_traj_num, prob_version, exp_type, role, scene_name, scene_num):
    start_length = 0
        
    if not os.path.exists(cliff_map_file):
        cliff_map_data = None
    else:
        cliff_map_data = utils.read_cliff_map_data(cliff_map_file)
        print("condition map_: ",cliff_map_file)
        
    if not os.path.exists(general_cliff_map_file):
        general_cliff_map_data = None
    else:
        general_cliff_map_data = utils.read_cliff_map_data(general_cliff_map_file)
        print("general map_: ",general_cliff_map_file)

    if cliff_map_data is None:
        pure_CVM = True

    human_traj_data = utils.read_role_sdd_human_traj_data(human_traj_file)

    person_id_list = utils.get_all_person_id(human_traj_data)

    header = ["person_id", "predict_horizon", "FDE_mean", "FDE_std", "FDE_min", "FDE_1", "FDE_median", "FDE_3", "ADE_mean", "ADE_std", "ADE_min", "ADE_1", "ADE_median", "ADE_3", "top_k_ADE_by_timestep", "num_predicted_trajs"]

    with open(result_file, "w", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
    
    for person_id in person_id_list:
        ################## For save plot of each prediction ###################################
        # plt.clf()
        # plt.close('all')
        # plt.figure(figsize=(15, 9), dpi=100)
        # img = plt.imread(f"mod_based_predictors/maps/sdd/annotations/{scene_name}/video{scene_num}/reference.jpg")
        # plt.imshow(img, origin='lower')
        # # plot_figures.plot_cliff_map(cliff_map_data)
        #######################################################################################
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
            dataset="sdd",
        )
        if not trajectory_predictor.check_human_traj_data(if_check=False):
            continue

        if pure_CVM:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_pure_cvm()
        else:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_stop(prob_version)

        ############ If fail to predict any of trajectory using MoDs, go to CVM instead ############
        if all_predicted_trajectory_list == []:
            all_predicted_trajectory_list = trajectory_predictor.predict_one_human_traj_pure_cvm()
        ############################################################################################
        
        ############ save one person's predicted trajectory ########################################
        # traj_save_dir = f"mod_based_predictors/results/sdd/{exp_type}_{scene_name}_{scene_num}_{role}"
        # utils.save_one_person_predicted_traj(all_predicted_trajectory_list, traj_save_dir)
        ############################################################################################

        if if_rank == 1:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_v3_rank(all_predicted_trajectory_list, version=2)
        elif if_rank == 0:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_v2(all_predicted_trajectory_list)
        elif if_rank == 2:
            predicted_horizon, res_FDE, res_ADE = trajectory_predictor.evaluate_ADE_FDE_result_most_likely(all_predicted_trajectory_list)

        ################## For save plot of each prediction ###################################
        # textstr = '\n'.join((
        #     "horizon = " + str(predicted_horizon),
        #     "ADE = " + "{:.3f}".format(round(res_ADE, 3)),
        #     "FDE = " + "{:.3f}".format(round(res_FDE, 3))))

        # props = dict(boxstyle='round', facecolor = "gainsboro", alpha=1)

        # plt.text(1500, 1000, textstr, fontsize=10,
        #         verticalalignment='top', bbox=props)
        # plt.title(f"{exp_type}_{scene_name}_{scene_num}_{role}_{person_id}")

        # plot_figures.plot_most_likely_trajectory(all_predicted_trajectory_list, observed_tracklet_length=observed_tracklet_length)
        # # plot_figures.plot_most_likely_traj_blue_colors(all_predicted_trajectory_list, observed_tracklet_length=observed_tracklet_length)
        # plot_figures.plot_human_traj(human_traj_data_by_person_id[start_length:, :], observed_tracklet_length=observed_tracklet_length)

        # fig_save_dir = f"mod_based_predictors/results/sdd_plots/{exp_type}_{scene_name}_{scene_num}_{role}_{person_id}"
        # os.makedirs(fig_save_dir, exist_ok=True)
        # plt.savefig(f"{fig_save_dir}/traj.png", bbox_inches='tight')
         
        # plt.show()
        ########################################################################################

  
def main(r_s, cliff_res, generate_traj_num, version, if_rank, beta, prob_version, exp_type=None):
    for test_per in [10]:
        for batch_num in range(1,11):
            sdd_folder = f"data/mod_based_predictors_data/sdd/k_fold_{test_per}percentage_test/batch{batch_num}/test"
            
            sdd_files = [f for f in os.listdir(sdd_folder) if os.path.isfile(os.path.join(sdd_folder, f))]
            
            for file_name in tqdm(sdd_files):
                role = file_name.split("_")[0]
                scene_name = file_name.split("_")[1]
                scene_num = file_name.split("_")[2].split(".")[0]
                scene = f"{scene_name}_{scene_num}"

                use_scene_ped = [
                    "deathCircle_0", "deathCircle_1", "deathCircle_3",
                    "bookstore_0", "bookstore_2", "bookstore_3", "bookstore_4", "bookstore_5",
                    "gates_0", "gates_1", "gates_2", "gates_3", "gates_4", "gates_5", "gates_7", "gates_8",
                    "little_0", "little_1", "little_2", "little_3",
                    "hyang_0", "hyang_1", "hyang_2", "hyang_3", "hyang_4", "hyang_5", "hyang_12",
                    "coupa_0",
                    "nexus_9"
                ]
                
                use_scene_biker = [
                    "deathCircle_0", "deathCircle_1", "deathCircle_3",
                    "bookstore_0", "bookstore_2", "bookstore_3", "bookstore_4", "bookstore_5",
                    "gates_0", "gates_1", "gates_2", "gates_3", "gates_4", "gates_5", "gates_7", "gates_8",
                    "little_0", "little_1", "little_2", "little_3",
                    "hyang_0", "hyang_1", "hyang_2", "hyang_3", "hyang_4", "hyang_5", "hyang_12",
                    "coupa_0"
                ]
                
                use_scene_car = ["nexus_9"]
                
                if role == "Pedestrian":
                    use_scene = use_scene_ped
                elif role == "Biker":
                    use_scene = use_scene_biker
                elif role == "Car":
                    use_scene = use_scene_car
                else:
                    continue
                
                if scene not in use_scene:
                    continue
                
                print(f"------------------in scene: {scene} {role} ------------------")
                general_cliff_map_file = f"mod_based_predictors/cliff/role_condition_k_fold_sdd/general_cluster_4_grid_{cliff_res}/k_fold_{test_per}percentage_test/batch{batch_num}/{scene}_cliff.csv"
                if exp_type == "condition":
                    cliff_map_file = f"mod_based_predictorscliff/role_condition_k_fold_sdd/role_cluster_4_grid_{cliff_res}/k_fold_{test_per}percentage_test/batch{batch_num}/{role}_{scene}_cliff.csv"                        
                elif exp_type == "general":
                    cliff_map_file = general_cliff_map_file

                human_traj_file = sdd_folder + "/" + file_name
                
                observed_tracklet_length = 7
                planning_horizon = 4.8
                pure_CVM = False
                delta_t = 0.4
                if_rank = if_rank
                
                result_folder = f"mod_based_predictors/results/sdd/{version}/k_fold_{test_per}percentage_test/batch{batch_num}"
                os.makedirs(result_folder, exist_ok=True)
                
                result_file = f"{result_folder}/{role}_{scene}.csv"
                
                run_experiment(cliff_map_file, general_cliff_map_file, human_traj_file, result_file, observed_tracklet_length, planning_horizon, delta_t, r_s, pure_CVM, if_rank, beta, generate_traj_num, prob_version, exp_type, role, scene_name, scene_num)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        r_s = float(sys.argv[1])
        cliff_res = sys.argv[2]
        generate_traj_num = int(sys.argv[3])
        version = sys.argv[4]
        if_rank = int(sys.argv[5])
        exp_type = sys.argv[6]
        beta = int(sys.argv[7])
        prob_version = 2

        print("We update params from args: ")
        print("The r_s is: ", r_s)
        print("The cliff_res is: ", cliff_res)
        print("The generate_traj_num is: ", generate_traj_num)
        print("The version is: ", version)
        print("The if_rank is: ", if_rank)
        print("The exp_type is: ", exp_type)
        
    else:
        r_s = 1
        generate_traj_num = 3 # the k value, like in top k
        version = "tmp"
        if_rank = False
        prob_version = 2
        exp_type = "condition"
    
    main(r_s, cliff_res, generate_traj_num, version, if_rank, beta, prob_version, exp_type)