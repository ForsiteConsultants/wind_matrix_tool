# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:30:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from windrose import WindroseAxes


def genWindMatrix(out_path: str,
                  ws_array: np.ndarray,
                  wd_array: np.ndarray,
                  ws_units: str,
                  ws_bin_width: int,
                  wd_bin_width: int,
                  dir_labels: list = None,
                  gen_wind_rose: bool = False) -> None:
    """
    Function to generate a wind matrix from wind speed and direction data
    :param out_path: path to save the wind matrix as a csv
    :param ws_array: an array of wind speed values (dimensions must match wd_array)
    :param wd_array: an array of wind direction values (dimensions must match ws_array)
    :param ws_units: the wind speed units - Options: ['kph', 'mph']
    :param ws_bin_width: the wind speed bin values
    :param wd_bin_width: the wind direction bin values
    :param dir_labels: the labels of the wind direction bin values
    :param gen_wind_rose: generate a wind rose figure from the data
    :return: None
    """
    # X-axis labels/bins (wind direction - degrees)
    if dir_labels is None:
        dir_labels = [x for x in range(0, 360 + wd_bin_width, wd_bin_width)]
    dir_bins = [int((x * wd_bin_width) / 2) for x in range(0, 1 + int(360 / wd_bin_width) * 2)]

    # Y-axis bins/labels (wind speed)
    if ws_units == 'mph':
        spd_bins = [x for x in range(0, 50 + ws_bin_width, ws_bin_width)]
    else:  # elif ws_units == 'kph'
        spd_bins = [x for x in range(0, 80 + ws_bin_width, ws_bin_width)]

    # Create empty array of zeros to store bin counts
    wind_array = np.zeros((len(spd_bins) - 1, len(dir_bins) - 2),
                          dtype=int)  # Create empty array of zeros to store bin counts

    # Generate wind histogram, and bin counts
    wnd_array = np.dstack((ws_array.flatten(),
                           wd_array.flatten()))  # Pair up wind speed/direction observations from same cells
    wnd_array = wnd_array[0][
        ~np.any(wnd_array[0] == 999, axis=1)]  # Remove cells where wind speed & direction = NoData (999)

    wnd_dir = [float(i[1]) for i in wnd_array]  # Re-separate wind direction values
    if ws_units == 'mph':
        wnd_spd = [0.6213711922 * float(i[0]) for i in
                   wnd_array]  # Re-separate wind speed values & convert km/hr to m/hr
    else:  # elif ws_units == 'kph'
        wnd_spd = [float(i[0]) for i in wnd_array]  # Re-separate wind speed values

    # Bin all wind speed/direction counts in histogram
    H, spd_bins, dir_bins = np.histogram2d(wnd_spd, wnd_dir, bins=((spd_bins, dir_bins)))
    # Add values from first column (22 degrees) to last column (360 degrees)
    H[:, -1] += H[:, 0]
    # Drop first column (22 degrees)
    H = np.delete(H, 0, 1)
    # Add new histogram counts to histogram counts in wind_array
    wind_array = np.add(wind_array, H)

    # Get total number of cells that were binned into windArray
    num_wnd_obs = np.sum(wind_array)
    # Convert windArray into probability matrix
    wind_matrix = 100 * wind_array / num_wnd_obs
    wind_matrix = wind_matrix[:len(spd_bins), :len(dir_labels) - 1]

    # Save the data to csv
    wind_df = pd.DataFrame(wind_matrix,
                           index=spd_bins[1:],
                           columns=dir_labels[1:])
    wind_df.to_csv(out_path, sep=',')

    if gen_wind_rose:
        genWindRose(wind_data=wind_df,
                    dir_bins=dir_labels[1:],
                    out_path=out_path.replace('.csv', '.png'))

    return


def genWindRose(wind_data: pd.DataFrame,
                ws_units: str,
                dir_bins: list,
                out_path: str) -> None:
    """
    Function to generate a wind rose from a wind matrix dataset with color gradient applied
    :param wind_data: the wind matrix data as a DataFrame
    :param ws_units: the wind speed units - Options: ['kph', 'mph']
    :param dir_bins: the list of direction bins (angles in degrees)
    :param out_path: path to save the wind rose figure
    :return: None
    """
    # Extract the wind speed and direction data for plotting
    speeds = wind_data.to_numpy().flatten()  # Flatten all speed values into a single array

    # Create the wind rose plot
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)

    # Create wind speed bins based on max wind speed (you can adjust these values based on your data)
    speed_bins = np.linspace(np.min(speeds), np.max(speeds), num=5)

    # Use a color map for the wind speed gradient (e.g., "viridis", "coolwarm")
    colormap = plt.cm.inferno  # You can choose another colormap like 'coolwarm', 'plasma', 'inferno', etc.

    # Plot the wind rose with the gradient applied based on speed bins
    ax.bar(dir_bins, speeds, normed=True, opening=0.8, edgecolor='white', bins=speed_bins, cmap=colormap)

    # Add legend for wind speed
    if ws_units == 'mph':
        ax.set_legend(title='Wind Speed (miles/hr)')
    else:  # elif ws_units == 'kph':
        ax.set_legend(title='Wind Speed (km/hr)')

    # Save the plot as an image file
    plt.savefig(out_path)
    plt.close()

    return
