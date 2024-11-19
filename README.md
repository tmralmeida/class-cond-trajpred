class-cond-trajpred
==============================

Implementation of "Trajectory Prediction for Heterogeneous Agents: A Performance Analysis on Small and Imbalanced Datasets".

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── original       <- Data to train the models
    │   ├── processed      <- Data to evaluate k-fold cross validation
    │
    ├── notebooks          <- Jupyter notebooks. 
    │
    ├── requirements.yml   <- The requirements file for reproducing the analysis environment
    │                         
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── class_cond_trajpred                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │   ├── mods         <- Scripts to train and evaluate MoD-based models
    │   │   │                 
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>



## Install

Install [miniconda](http://docs.conda.io/en/latest/miniconda.html). Then, you can install all packages required by running:

```
conda env create -f requirements.yml && conda activate class-cond-trajpred
```

And run:
```
pip install .
```

## Training and testing deep learning predictors

Steps to train and test deep learning-based predictors (i.e., RED, cRED, TF, cTF, GAN, cGAN, VAE, and cVAE).

### Train one model on a dataset

Set the corresponding cfg file on  the [config folder](https://github.com/tmralmeida/class-cond-trajpred/tree/main/class_cond_trajpred/cfg) to the original data as follows:

------------
    data :
      data_dir : data/original/thor_magni/  # data/original/sdd/ for SDD dataset
------------

Run the train_model module with the corresponding config file:
```
python -m class_cond_trajpred.data_modeling.train_model class_cond_trajpred/cfg/thor_magni/sup_ctf.yaml 
```

### k-fold cross validation

Set the corresponding cfg file on  the [config folder](https://github.com/tmralmeida/class-cond-trajpred/tree/main/class_cond_trajpred/cfg) to the processed data as follows:

------------
    data :
      data_dir : data/processed/sdd/k_fold_10  # data/original/thor_magni/ for THOR-MAGNI dataset
------------


```
python -m class_cond_trajpred.data_modeling.k_fold_cv 10 class_cond_trajpred/cfg/thor_magni/gan.yaml
```

To train with `(x, y, v_x, v_y)`, we need to change the cfg file under `network`: set `observation_len` to 8.



## Training and testing MoD-based predictors
In MoD-based predictors, we have class-conditioned CLiFF-LHMP and general CLiFF-LHMP. Both methods use same evaluation datasets: SDD and THÖR-MAGNI. The corresponding dataset directories are located at `data/mod_based_predictors_data/sdd` and  `data/mod_based_predictors_data/thor_magni`. Within the dataset folder, test ratios ranging from 10% to 90% are available. For instance, `k_fold_10_percentage_test` corresponds to configurations where 10% of the data is used for testing and the remaining 90% for training. In each test ratio configuration, the test data are repeatedly and randomly sub-sampled 10 times, resulting in batches from `batch1` to `batch10`.

To run MoD predictors:
```
./mod_based_predictors/run_mod_predictors.sh
```
In `run_mod_predictors.sh`, it contains command to run MoD predictors for both datasets, take class-conditioned CLiFF-LHMP for THÖR-MAGNI dataset as an example:
```
r_s=0.2
exp_type="condition" 
if_rank=2
generate_traj_num=5
beta=1

version="r${r_s}_k${generate_traj_num}_${exp_type}"
python3 mod_based_predictors/main_mod_magni.py "$r_s" "$generate_traj_num" "$version" "$if_rank" "$exp_type" "$beta"
```

Here, the input parameters are:
- r_s: sampling radius, for sampling velocity in MoD
- exp_type: "condition" or "general"
- if_rank: 
  - 0: for evaluate with k predicted trajectory, output average ADE/FDE
  - 1: for evaluate with k predicted trajectory, output weighted average ADE/FDE
  - 2: for evaluate with most-likely predicted trajectory,

- generate_traj_num: generate k predicted trajectories
- beta: for controlling the reliance on the MoD versus the CVM,with a lower β favoring the velocity sampled from the MoD.

The output evaluation metrics are saved in `mod_based_predictors/results`. It is also optional to save and plot predicted trajectories.



