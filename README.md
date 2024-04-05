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



## Installation


Conda environemnt:
```

```



```
pip install .
```




## Training and testing deep learning predictors

### Train one model on a dataset


Set the corresponding cfg file on  the [cfg folder](https://github.com/tmralmeida/class-cond-trajpred) to the original data as follows:
------------
data :
  data_dir : data/original/thor_magni/  # data/original/sdd/ for SDD dataset
------------

Run the train_model module with the corresponding config file:
```
python -m class_cond_trajpred.data_modeling.train_model class_cond_trajpred/cfg/thor_magni/sup_ctf.yaml 
```

### k-fold cross validation

Set the corresponding cfg file on  the [cfg folder](https://github.com/tmralmeida/class-cond-trajpred) to the processed data as follows:
------------
data :
  data_dir : data/processed/sdd/k_fold_10  # data/original/thor_magni/ for THOR-MAGNI dataset
------------

```
python -m src.data_modeling.k_fold_cv 10 src/cfg/models/deep_learning_based/thor_magni/forecasting/rnn.yaml
```

To train with `(x, y, v_x, v_y)`, we need to change the cfg file under `network`: set `observation_len` to 8.



## Training and testing MoD-based predictors