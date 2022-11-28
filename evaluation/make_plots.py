# The purpose of this script is to create
# roc plots from data saved in files (first execute roc.py)
import os

import matplotlib.pyplot as plt
import numpy as np


def read_and_plot(subDir, sensitivity_file, precision_file, fieldpoint_type):
    """read sensitivity, precision from files and plot against each other

    Args
        subdir: directory containing sensitivity and precision files
        sensitivity_file: numpy file containing sensitivity 
        precision_file: numpy file containing precision
        fieldpoint_type: type of fieldpoint, encoded as follows
            "5": "Negative Electrostatic",
            "6": "Positive Electrostatic",
            "7": "van der Waals",
            "8": "Hydrophobic",
    """
    # read from file and generate plots
    file_dir = os.path.dirname(os.path.realpath(__file__))
    sensitivity = np.loadtxt(file_dir + "/images/" + subDir + sensitivity_file)
    precision = np.loadtxt(file_dir + "/images/" + subDir + precision_file)
    filterLevels = np.loadtxt(file_dir + "/images/" + subDir + filter_level_file)
    # filter for nans
    isNanSensitivity = np.isnan(sensitivity)
    isNanPrecision = np.isnan(precision)
    indexFilter = np.logical_and(
        np.logical_not(isNanSensitivity), np.logical_not(isNanPrecision)
    )
    sensitivityFiltered = sensitivity[indexFilter]
    precisionFiltered = precision[indexFilter]
    filter_levels_filtered = filterLevels[indexFilter]
    plot_sens_vs_prec(
        precisionFiltered,
        sensitivityFiltered,
        subDir,
        fieldpoint_type,
        filter_levels_filtered,
    )


def plot_sens_vs_prec(
    precision, sensitivity, save_plot_path, fieldpoint_type, filter_levels_filtered
):
    """Plot weighted sensitivity vs precision

    Args
        precision: numpy array of precision
        sensitivity: numpy array of sensitivity
        save_plot_path: path to save the plot
        fieldpoint_type: type of field point encoded as stringified number, e.g. "5"
        filter_levels_filtered: filter levels (but only those kept which produce values different from None)
    """
    fieldpoint_types = {
        "5": "Negative Electrostatic",
        "6": "Positive Electrostatic",
        "7": "van der Waals",
        "8": "Hydrophobic",
    }
    fieldpoint_typeName = fieldpoint_types[str(fieldpoint_type)]
    plt.title(f"{fieldpoint_typeName}")
    plt.plot(precision, sensitivity, c="C0")
    plt.ylabel("$L_{WTPR}^{0.5}$", fontsize=12)
    plt.xlabel(r"$L_{PPV}^{0.5}$", fontsize=12)
    # highlight points in the graph for certain filter levels: minmal,maximal, and 0.005)
    plt.scatter(precision[0], sensitivity[0], marker="o", s=10, c="#1f77b4")
    plt.annotate(
        "c={:.4f}".format(filter_levels_filtered[0]),  # this is the text
        (
            precision[0],
            sensitivity[0],
        ),  # these are the coordinates to position the label
        textcoords="offset points",  # how to position the text
        xytext=(-3, 0),  # distance from text to points (x,y)
        ha="right",  # horizontal alignment can be left, right or center
        fontsize="small",
    )
    plt.scatter(precision[-3], sensitivity[-3], marker="o", s=10, c="#1f77b4")
    plt.annotate(
        "c={:.4f}".format(filter_levels_filtered[-3]),  # this is the text
        (
            precision[-3],
            sensitivity[-3],
        ),  # these are the coordinates to position the label
        textcoords="offset points",  # how to position the text
        xytext=(0, 5),  # distance from text to points (x,y)
        ha="left",  # horizontal alignment can be left, right or center
        fontsize="small",
    )
    plt.scatter(precision[-1], sensitivity[-1], marker="o", s=10, c="#1f77b4")
    plt.annotate(
        "c={:.4f}".format(filter_levels_filtered[-1]),  # this is the text
        (
            precision[-1],
            sensitivity[-1],
        ),  # these are the coordinates to position the label
        textcoords="offset points",  # how to position the text
        xytext=(0, 3),  # distance from text to points (x,y)
        ha="left",  # horizontal alignment can be left, right or center
        fontsize="small",
    )
    file_dir = os.path.dirname(os.path.realpath(__file__))
    plt.savefig(
        file_dir
        + "/images/"
        + save_plot_path
        + f"fieldpoint_{np.abs(fieldpoint_type)}_plot.pdf"
    )
    plt.close()


if __name__ == "__main__":
    subDir = "shortversion/"
    fieldpoint_types = [5, 6, 7, 8]
    for fieldpoint_type in fieldpoint_types:
        # sensitivity_file = f"shortVersionfieldpoint_{fieldpoint_type}_r1_sensitivity"
        # precision_file = f"shortVersionfieldpoint_{fieldpoint_type}_r1_precision"
        # filter_level_file = f"shortVersionfieldpoint_{fieldpoint_type}_r1_filterLevels"
        sensitivity_file = f"sortVersionfieldpoint_{fieldpoint_type}_sensitivity"
        precision_file = f"sortVersionfieldpoint_{fieldpoint_type}_precision"
        filter_level_file = f"sortVersionfieldpoint_{fieldpoint_type}_filterLevels"
        read_and_plot(subDir, sensitivity_file, precision_file, fieldpoint_type)

