from typing import Optional
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_cmap(n, name="hsv"):
    """Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name."""
    cmap = plt.get_cmap(name)
    values = np.linspace(0, 1, n)
    return [cmap(value) for value in values]


def visualize_dataset(
    trajectories: pd.DataFrame, combination_list: dict, skip_window_size: int
):
    """visualize trajectories of a ordered df"""
    cmap = get_cmap(len(combination_list))
    plt.figure(figsize=(20, 10))
    for i in range(0, len(trajectories), skip_window_size):
        traj = trajectories.iloc[i : i + skip_window_size]
        row = traj.iloc[0]
        data_label = f"{row['ag_id']},{row['data_label']}"
        plt.plot(
            traj["x"],
            traj["y"],
            label=data_label,
            color=cmap[combination_list.index(data_label)],
        )
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.plot()


def visualize_by_role(
    dataset: str,
    df: pd.DataFrame,
    skip_window_size: int,
    x_col: str = "x",
    y_col: str = "y",
    map_layout: Optional[np.array] = None,
) -> dict:
    if x_col == "x" and y_col == "y" and dataset == "thor":
        x_limits, y_limits = (-6, 7), (-7, 10)
    elif x_col == "x" and y_col == "y" and dataset == "thor_magni":
        x_limits, y_limits = (-10, 10), (-5, 4)
    elif x_col == "x_speed" and y_col == "y_speed":
        x_limits, y_limits = (-2.5, 2.5), (-2.5, 2.5)
    elif x_col == "x_pix" and y_col == "y_pix":
        x_limits, y_limits = (0, 2300), (0, 2350)
    f, axes = plt.subplots(
        math.ceil(len(df.data_label.unique()) / 3),
        3
    )
    agents_in_scene = df.data_label.unique()
    trajectories_counter = {}
    if map_layout is not None:
        for i, ax in enumerate(axes.flatten()):
            if i >= len(agents_in_scene):
                break
            ax.imshow(map_layout)
    for i, ax in enumerate(axes.flatten()):
        if i >= len(agents_in_scene):
            break
        data_label = agents_in_scene[i]
        trajectories_counter[data_label] = 0
        target_agent = df[df.data_label == data_label]
        ax.set_title(data_label)
        for j in range(0, len(target_agent), skip_window_size):
            trajectories_counter[data_label] += 1
            traj = target_agent.iloc[j : j + skip_window_size]
            ax.plot(traj[x_col], traj[y_col])
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        if x_col == "x_pix":
            ax.set_axis_off()
    f.tight_layout()
    return trajectories_counter


def get_multi_label_combination(df: pd.DataFrame):
    """main label stored at ag_id and sub label saved at ag_id"""
    unique_combinations = df[["ag_id", "data_label"]].drop_duplicates()
    combination_list = unique_combinations.apply(
        lambda row: ",".join(row.values.astype(str)), axis=1
    ).tolist()
    return combination_list
