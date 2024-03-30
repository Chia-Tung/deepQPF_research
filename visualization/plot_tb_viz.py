import ast

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.const import CONFIG, COUNTY_DATA, CWBRR, NORM


def plot_data(ax, data):
    # get constant
    dl_config = CONFIG["train_config"]["data_loader_params"]
    lat_s = dl_config["target_lat"][0]
    lat_e = dl_config["target_lat"][1]
    lon_s = dl_config["target_lon"][0]
    lon_e = dl_config["target_lon"][1]
    target_shp = ast.literal_eval(dl_config["target_shape"])

    # model axis
    lat = np.linspace(lat_s, lat_e, target_shp[0])
    lon = np.linspace(lon_s, lon_e, target_shp[1])
    lon, lat = np.meshgrid(lon, lat)

    # plot background
    ax = COUNTY_DATA.plot(
        ax=ax, color="none", edgecolor="black", linewidth=0.1, zorder=1
    )

    # plot data
    ax.pcolormesh(
        lon,
        lat,
        data,
        edgecolors="none",
        shading="auto",
        norm=NORM,
        cmap=CWBRR,
        zorder=0,
    )

    # canvas setting
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(lon_s, lon_e)
    ax.set_ylim(lat_s, lat_e)
    return ax


def gen_plot(data: np.ndarray | torch.Tensor):
    """
    data shape is either [S, C, H, W] or [S, H, W]
    """
    if isinstance(data, torch.Tensor):
        data = data.cpu()

    columns = data.shape[0]
    if len(data.shape) == 3:
        rows = 1

        fig, ax = plt.subplots(rows, columns, figsize=(6, 2), dpi=200, facecolor="w")
        if columns == 1:
            plot_data(ax, data[0])
        else:
            for i in range(columns):
                plot_data(ax[i], data[i])

    elif len(data.shape) == 4:
        rows = data.shape[1]

        fig, ax = plt.subplots(rows, columns, figsize=(6, 2), dpi=200, facecolor="w")
        for i in range(rows):
            for j in range(columns):
                plot_data(ax[i, j], data[j, i])

    return fig
