# The purpose of this script is the generation of
# plots comparing weighted sensitivity and precision for different descriptors and setups per field point
import copy

import matplotlib.pyplot as plt
import numpy as np


def create_plots(
    sensitivity,
    precision,
    color_list,
    markers,
    indexList,
    title,
    plot_name,
    labels,
    base_model_label=None,
):
    """Create sensitivity vs precision plots
    
    Args:
        sensitivity: numpy array of weighted sensitivy
        precision: numpy array of precision
    """
    labels_intern = copy.deepcopy(labels)
    if base_model_label != None:
        labels_intern[0] = base_model_label

    size = 30
    for ind in indexList:
        plt.scatter(
            sensitivity[ind],
            precision[ind],
            color=color_list[ind],
            marker=markers[ind],
            s=size,
        )
    plt.title(title, fontsize=12)
    plt.xlabel("$L_{WTPR}^{0.5}$", fontsize=12)
    plt.ylabel("$L_{PPV}^{0.5}$", fontsize=12)
    plt.legend(labels_intern[indexList])
    plt.tight_layout()
    plt.savefig("./images/model_performance/" + plot_name + ".pdf", format="pdf")
    plt.close()


if __name__ == "__main__":
    # data for 0.5 Angstroem
    # ordering: full model 7A, full model 5A, covalent graph topology, atom type, node degree, atom size,
    #           partial charge, logP, without atom type, without node degree, without atom size, without partial charge,
    #           without logP,
    # negative field point:
    sensitivity_negative_fieldpoint = np.array(
        [
            0.848,
            0.821,
            0.536,
            0.814,
            0.786,
            0.786,
            0.805,
            0.778,
            0.835,
            0.852,
            0.843,
            0.821,
            0.842,
        ]
    )
    precision_negative_fieldpoint = np.array(
        [
            0.755,
            0.711,
            0.349,
            0.701,
            0.677,
            0.654,
            0.708,
            0.666,
            0.737,
            0.745,
            0.744,
            0.723,
            0.750,
        ]
    )
    # positive field point:
    sensitivity_positive_fieldpoint = np.array(
        [
            0.837,
            0.832,
            0.343,
            0.818,
            0.765,
            0.779,
            0.822,
            0.801,
            0.829,
            0.865,
            0.848,
            0.811,
            0.836,
        ]
    )
    precision_positive_fieldpoint = np.array(
        [
            0.825,
            0.788,
            0.258,
            0.760,
            0.748,
            0.744,
            0.760,
            0.729,
            0.801,
            0.788,
            0.803,
            0.777,
            0.811,
        ]
    )
    # hydrophobic field point:
    sensitivity_hydro_fieldpoint = np.array(
        [
            0.960,
            0.958,
            0.871,
            0.966,
            0.949,
            0.953,
            0.950,
            0.952,
            0.963,
            0.962,
            0.955,
            0.957,
            0.956,
        ]
    )
    precision_hydro_fieldpoint = np.array(
        [
            0.906,
            0.897,
            0.728,
            0.872,
            0.881,
            0.894,
            0.889,
            0.878,
            0.898,
            0.903,
            0.903,
            0.887,
            0.906,
        ]
    )
    # vdw field point:
    sensitivity_vdw_fieldpoint = np.array(
        [
            0.820,
            0.811,
            0.309,
            0.822,
            0.795,
            0.789,
            0.770,
            0.794,
            0.787,
            0.803,
            0.824,
            0.819,
            0.823,
        ]
    )
    precision_vdw_fieldpoint = np.array(
        [
            0.878,
            0.832,
            0.332,
            0.845,
            0.810,
            0.817,
            0.811,
            0.810,
            0.857,
            0.823,
            0.874,
            0.853,
            0.857,
        ]
    )
    #####################################
    # Plot settings:
    #####################################
    # diferent colour for each experimental setup
    colour_list = ["black", "blue", "brown"]
    colour_list.extend(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    # diferent symbol for each experimental setup
    markers = ["x", "v", "^", "4", ">", "2", "p", "3", "*", ".", "+", "D", "1"]
    size = 20
    # label for each experimental setup
    labels = np.array(
        [
            "all predictors",
            "$5\, \AA$ distance topology",
            "covalent bond graph topology",
            'only "atom type"',
            'only "node degree"',
            'only "atom size"',
            'only "partial charge"',
            'only "W.-C. logP"',
            'leave out "atom type"',
            'leave out "node degree"',
            'leave out "atom size"',
            'leave out "partial charge"',
            'leave out "W.-C. logP"',
        ]
    )
    #####################################
    # negative field points:
    #####################################
    # leave one predictor out:
    indexList = [0, 8, 9, 10, 11, 12]
    create_plots(
        sensitivity_negative_fieldpoint,
        precision_negative_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Negative Fieldpoints",
        "negFp_leave_out",
        labels,
    )
    # only single predictors:
    indexList = [0, 3, 4, 5, 6, 7]
    create_plots(
        sensitivity_negative_fieldpoint,
        precision_negative_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Negative Fieldpoints",
        "negFp_single_predictors",
        labels,
    )
    # different graph topologies:
    indexList = [0, 1, 2]
    create_plots(
        sensitivity_negative_fieldpoint,
        precision_negative_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Negative Fieldpoints",
        "negFp_topology_comparison",
        labels,
        base_model_label="$7\, \AA$ distance topology",
    )
    #####################################
    # positive field points
    #####################################
    # leave one predictor out:
    indexList = [0, 8, 9, 10, 11, 12]
    create_plots(
        sensitivity_positive_fieldpoint,
        precision_positive_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Positive Fieldpoints",
        "posFp_leave_out",
        labels,
    )
    # only single predictors:
    indexList = [0, 3, 4, 5, 6, 7]
    create_plots(
        sensitivity_positive_fieldpoint,
        precision_positive_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Positive Fieldpoints",
        "posFp_single_predictors",
        labels,
    )
    # different graph topologies:
    indexList = [0, 1, 2]
    create_plots(
        sensitivity_positive_fieldpoint,
        precision_positive_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Positive Fieldpoints",
        "posFp_topology_comparison",
        labels,
        base_model_label="$7\, \AA$ distance topology",
    )
    #####################################
    # vdw
    #####################################
    # leave one predictor out:
    indexList = [0, 8, 9, 10, 11, 12]
    create_plots(
        sensitivity_vdw_fieldpoint,
        precision_vdw_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Van der Waals Fieldpoints",
        "vdwFp_leave_out",
        labels,
    )
    # only single predictors:
    indexList = [0, 3, 4, 5, 6, 7]
    create_plots(
        sensitivity_vdw_fieldpoint,
        precision_vdw_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Van der Waals Fieldpoints",
        "vdwFp_single_predictors",
        labels,
    )
    # different graph topologies:
    indexList = [0, 1, 2]
    create_plots(
        sensitivity_vdw_fieldpoint,
        precision_vdw_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Van der Waals Fieldpoints",
        "vdwFp_topology_comparison",
        labels,
        base_model_label="$7\, \AA$ distance topology",
    )
    #####################################
    # hydrophobic
    #####################################
    # leave one predictor out:
    indexList = [0, 8, 9, 10, 11, 12]
    create_plots(
        sensitivity_hydro_fieldpoint,
        precision_hydro_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Hydrophobic Fieldpoints",
        "hydroFp_leave_out",
        labels,
    )
    # only single predictors:
    indexList = [0, 3, 4, 5, 6, 7]
    create_plots(
        sensitivity_hydro_fieldpoint,
        precision_hydro_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Hydrophobic Fieldpoints",
        "hydroFp_single_predictors",
        labels,
    )
    # different graph topologies:
    indexList = [0, 1, 2]
    create_plots(
        sensitivity_hydro_fieldpoint,
        precision_hydro_fieldpoint,
        colour_list,
        markers,
        indexList,
        "Model Performance: Hydrophobic Fieldpoints",
        "hydroFp_topology_comparison",
        labels,
        base_model_label="$7\, \AA$ distance topology",
    )
