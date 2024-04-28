#!/bin/bash

# ############### For SDD class-conditioned MoD predictor ###############
# r_s=20
# exp_type="condition"
# if_rank=2
# generate_traj_num=5
# beta=5
# cliff_res=20

# version="r${r_s}_k${generate_traj_num}_${exp_type}_beta${beta}_cliff_cluster_4_grid_${cliff_res}"
# python3 mod_based_predictors/main_mod_sdd.py "$r_s" "$cliff_res" "$generate_traj_num" "$version" "$if_rank" "$exp_type" "$beta"

# ############### For SDD general MoD predictor ###############
# r_s=20
# exp_type="general"
# if_rank=2
# generate_traj_num=5
# beta=5
# cliff_res=20

# version="r${r_s}_k${generate_traj_num}_${exp_type}_beta${beta}_cliff_cluster_4_grid_${cliff_res}"
# python3 mod_based_predictors/main_mod_sdd.py "$r_s" "$cliff_res" "$generate_traj_num" "$version" "$if_rank" "$exp_type" "$beta"



############### For MAGNI class-conditioned MoD predictor ###############
r_s=0.2
exp_type="condition"
if_rank=2
generate_traj_num=5
beta=1

version="r${r_s}_k${generate_traj_num}_${exp_type}"
python3 mod_based_predictors/main_mod_magni.py "$r_s" "$generate_traj_num" "$version" "$if_rank" "$exp_type" "$beta"

# ############### For MAGNI general MoD predictor ###############
# r_s=0.2
# exp_type="general"
# if_rank=2
# generate_traj_num=5
# beta=1

# version="r${r_s}_k${generate_traj_num}_${exp_type}"
# python3 mod_based_predictors/main_mod_magni.py "$r_s" "$generate_traj_num" "$version" "$if_rank" "$exp_type" "$beta"
