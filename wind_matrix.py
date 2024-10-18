# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 16:30:00 2024

@author: Gregory A. Greene
"""
__author__ = ['Gregory A. Greene, map.n.trowel@gmail.com']

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.colors import LinearSegmentedColormap


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

    # Round all values to the nearest 0.01
    wind_matrix = np.round(wind_matrix, 2)

    # Save the data to csv
    wind_df = pd.DataFrame(wind_matrix,
                           index=spd_bins[1:],
                           columns=dir_labels[1:])
    wind_df.to_csv(out_path, sep=',')

    if gen_wind_rose:
        genWindRose(wind_data=wind_df,
                    ws_units=ws_units,
                    dir_bins=dir_labels[1:],
                    out_path=out_path.replace('.csv', '.png'))

    return


def _adjust_colormap_lightness(cmap: LinearSegmentedColormap,
                               factor: float = 0.85) -> LinearSegmentedColormap:
    """
    Adjust the lightness of the colormap by multiplying the lightness values by the factor.
    A factor < 1 will darken the colormap, and a factor > 1 will lighten it.

    :param cmap: The input colormap (matplotlib cmap)
    :param factor: The lightness factor to apply (default: 0.5 to make it lighter)
    :return: A new colormap with adjusted lightness
    """
    cmap = plt.get_cmap(cmap)

    # Create an array with the original colormap colors
    colors = cmap(np.linspace(0, 1, 256))

    # Adjust the lightness by applying the factor
    adjusted_colors = np.clip(colors[:, :3] * factor, 0, 1)  # Multiply RGB values, keep them between 0 and 1

    # Create a new colormap with adjusted colors
    new_cmap = mcolors.LinearSegmentedColormap.from_list(f'{cmap.name}_adjusted', adjusted_colors)

    return new_cmap


def genWindRose(wind_data: pd.DataFrame,
                ws_units: str,
                dir_bins: list,
                out_path: str = None,
                colormap: str = 'Blues',
                lightness_factor: float = 0.85) -> None:
    """
    Function to generate a wind rose from a wind matrix dataset with custom polar plotting.
    Wind data should be in the format where rows are wind speeds and columns are wind directions..
    Rows with no values greater than 0 are trimmed before plotting.

    :param wind_data: the wind matrix data as a DataFrame (values are proportions of wind observations)
    :param ws_units: the wind speed units - Options: ['kph', 'mph']
    :param dir_bins: the list of direction bins (angles in degrees)
    :param out_path: path to save the wind rose figure (optional)
    :param colormap: the colormap to use for the wind rose (default: 'viridis')
    :param lightness_factor: factor to adjust the lightness of the colormap (default: 1.5 to lighten)
    :return: None
    """
    # Remove rows where all values are zero
    wind_data = wind_data[(wind_data > 0).any(axis=1)]

    # Extract wind speeds (row indices) and the wind percentage matrix
    wind_speeds = wind_data.index.to_numpy()  # Wind speeds (row names or index)
    wind_percentages = wind_data.to_numpy()  # Wind observations

    # Ensure the wind data rows match the number of directions
    if wind_percentages.shape[1] != len(dir_bins):
        raise ValueError(f'Shape mismatch: wind data has {wind_percentages.shape[1]} columns, '
                         f'but {len(dir_bins)} directions were provided.')

    # Convert directions to radians for polar plot
    directions_rad = np.radians(dir_bins)

    # Dynamically calculate the width of each bar based on the number of directions
    bar_width = (2 * np.pi / len(dir_bins)) - 0.05 * (2 * np.pi / len(dir_bins)) # The total circle is 2π radians

    # Create a colormap for wind speeds
    cmap = _adjust_colormap_lightness(colormap, factor=lightness_factor)  # Adjust the lightness of the colormap
    num_speeds = len(wind_speeds)
    colors = cmap(np.linspace(0, 1, num_speeds))  # Assign color to each wind speed bin

    # Create a polar plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    # Lighten radial and angular gridlines by adjusting their color and alpha
    ax.grid(True, which='both', axis='both', color='gray', alpha=0.1, linewidth=0.7, zorder=0)

    # Initialize the bottom array for stacking the bars
    bottom = np.zeros(len(dir_bins))  # Start at the bottom (0) for the first layer

    # Plot the wind percentages for each speed category with consistent color for each wind speed
    for i, wind_speed in enumerate(wind_speeds):
        ax.bar(directions_rad, wind_percentages[i], width=bar_width, bottom=bottom, alpha=0.8, color=colors[i],
               zorder=1, label=f'{int(wind_speed)} {ws_units}')  # Bars with zorder=2 (higher than gridlines)
        # Update the bottom for the next layer (stack the bars)
        bottom += wind_percentages[i]

    # Set the direction labels and make sure they are in front (zorder=3)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_theta_offset(np.pi / 2.0)  # Set North (90°) to the top
    ax.set_xticks(directions_rad)
    ax.set_xticklabels([f'{int(d)}°' for d in dir_bins], zorder=3)  # Labels with zorder=3 (in front)

    # Set the radial (y-axis) labels in front with zorder=3
    ax.set_rlabel_position(0)  # Position radial labels at 0 degrees (to the right)
    ax.yaxis.set_tick_params(labelsize=10, zorder=3)  # Radial labels with zorder=3 (in front of the bars)

    # Extend the radial axis limit to the nearest multiple of 2 above the maximum value
    max_value = np.max(bottom)  # Get the maximum value from the stacked bars
    radial_limit = np.ceil(max_value / 2) * 2  # Round up to the nearest multiple of 2
    ax.set_ylim(0, radial_limit)  # Set the limit of the radial axis

    # Customize the radial (y-axis) labels to include "%" symbol
    radial_ticks = np.linspace(0, radial_limit, num=5)  # Explicitly set radial ticks
    ax.set_yticks(radial_ticks)  # Set the y-ticks explicitly
    radial_labels = [f'{round(tick, 1)}%' for tick in radial_ticks]  # Add "%" to each tick
    ax.set_yticklabels(radial_labels)  # Set the new y-tick labels with "%"

    # Add legend and title without zorder in the legend
    ax.legend(title=f'Wind Speed ({ws_units})', bbox_to_anchor=(1.1, 1.05))
    ax.set_title('Proportion of Wind Observations by Speed and Direction', zorder=3)

    # Save or display the plot
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

    return
