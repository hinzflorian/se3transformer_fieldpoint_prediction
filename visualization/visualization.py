"""In this script we write the visualization function"""
import math
import os
import random
import shutil

import numpy as np
import torch
from runtime.clustering import merge_clusters
from sklearn.cluster import AgglomerativeClustering


def write_xed(
    edges,
    atom_data,
    fieldpoint_data,
    predicted_fp=None,
    predicted_weights=None,
    xed_path=None,
):
    """Write xed file

    Args:
        edges: edges of the molecule
        atom_data: 
        fieldpoint_data: data of fieldpoint (location, type, value)
        predicted_fp: predicted location of fieldpoint 
        predicted_weights: predicted weight of field point
        xed_path: file path to save xed file
    """
    src = edges[0] + 1
    dst = edges[1] + 1
    src = list(src)
    dst = list(dst)
    nr_edges = len(src)
    nr_atoms = atom_data.shape[0]
    nr_fieldpoints = len(fieldpoint_data)
    if predicted_fp is None:
        nr_fieldpoints_predicted = 0
    else:
        nr_fieldpoints_predicted = len(predicted_fp)
    first_line = (
        f"2.185 {nr_atoms+nr_fieldpoints+nr_fieldpoints_predicted} {nr_edges} v1.0F3\n"
    )
    edge_lines = []
    edge_line = []
    for ind in range(len(src)):
        edge_line.append(str(int(src[ind])))
        edge_line.append(" ")
        edge_line.append(str(int(dst[ind])))
        edge_line.append(" ")
        if (ind + 1) % 5 == 0 or ind == len(src) - 1:
            edge_line = "".join(edge_line)
            edge_lines.append(edge_line)
            edge_line = []
    if xed_path == None:
        file_dir, _ = os.path.split(__file__)
        xed_path = file_dir + "/visualizer/visualization.xed"
    with open(xed_path, "w") as xed_file:
        # write first line
        xed_file.write(first_line + "\n")
        for line in edge_lines:
            xed_file.write(line + "\n")
        # write atom_data
        atom_data_write = atom_data[:, :6]
        for row in atom_data_write:
            row_convert = [int(row[0]), row[1], row[2], row[3], int(row[4]), row[5]]
            line = " ".join(str(entry) for entry in row_convert)
            xed_file.write(line + "\n")
        # write fieldpoint_data (true fieldpoints)
        # always use fieldpoint type -7 for true fieldpoint
        # and -6 for predictions
        fieldpoint_data[:, 0] = -7
        for row in fieldpoint_data:
            row_convert = [int(row[0]), row[1], row[2], row[3], int(row[4]), row[5]]
            line = " ".join(str(entry) for entry in row_convert)
            xed_file.write(line + "\n")
        # write the pedictions
        for ind in range(nr_fieldpoints_predicted):
            fp_coord = predicted_fp[ind]
            fp_size = predicted_weights[ind]
            line = " ".join(
                [
                    "-6",
                    str(fp_coord[0]),
                    str(fp_coord[1]),
                    str(fp_coord[2]),
                    str(fp_size),
                ]
            )
            xed_file.write(line + "\n")


def random_prediction(model, dataset, sample_index=None):
    """Make predictions on dataset[sample_index]

    Args:
        model: the (trained) SE(3)-Transformer model
        dataset: the data set to get samples 
        sample_index: index of sample in data set to make predictions on

    Return:
    """
    if sample_index == None:
        num_samples = len(dataset)
        random.seed(10)
        sample_index = random.randint(0, num_samples - 1)
        print(
            f"Sample Number: {sample_index} of {num_samples} in total ({sample_index/num_samples*100}%)"
        )
    edges, atom_data, _, fieldpoint_data, _ = dataset.get_item_data(sample_index)
    g, _, _ = dataset[sample_index]
    model = model.to("cpu")
    prediction = model(g, {"0": g.ndata["node_feats"]}, None)
    prediction = [val for _, val in prediction.items()]
    predicted_vectors = prediction[1].detach().numpy()
    predicted_probs = (
        torch.nn.Softmax(dim=0)(prediction[0].reshape((-1))).detach().numpy()
    )
    predicted_fp = atom_data[:, 1:4].reshape((-1, 1, 3)) + predicted_vectors
    predicted_fp = predicted_fp.reshape((-1, 3))
    relevant_preds = (
        predicted_probs > 0.0001
    )  # only keep predictions larger than a threshold
    predicted_probs = predicted_probs[relevant_preds]
    predicted_fp = predicted_fp[relevant_preds]
    return predicted_probs, predicted_fp, edges, atom_data, fieldpoint_data


def write_cluster_xed(
    cluster_centers,
    cluster_center_probs,
    atom_data,
    edges,
    fieldpoint_data,
    filter_level,
    xed_path_clustering,
):
    """Write xed file with clustered predictions of field point of molecule

    Args:
        cluster_centers: centers of clusters
        cluster_center_probs: sum of prob. of predictions belonging to cluster
        atom_data: 
        edges: edges of molecule
        fieldpoint_data: 
        filter_level: keep cluster if cluster probability is larger than this threshold
        xed_path_clustering: path, where to save the xed file
    """
    relevant_idx = cluster_center_probs > filter_level
    cluster_center_probs = cluster_center_probs[relevant_idx]
    cluster_centers = cluster_centers[relevant_idx]
    write_xed(
        edges,
        atom_data,
        fieldpoint_data,
        cluster_centers,
        cluster_center_probs,
        xed_path_clustering,
    )


def write_ranomd_pymols(
    dataset,
    model,
    xed_path_cloud,
    mol2_path_cloud,
    pymol_file_cloud,
    xed_path_clustering,
    mol2_path_clustering,
    pymol_file_clustering,
    cutoff,
    dst_threshold,
    sample_index,
):
    """Choose random samples from dataset, make predictions and write xed and pymol files

    Args:
        dataset: the data set to get samples 
        model: the (trained) SE(3)-Transformer model
        xed_path_cloud: path to xed file saving the molecule with unclustered field point predictions
        mol2_path_cloud: path to mol2 file saving the molecule with unclustered field point predictions
        pymol_file_cloud: path to pymol file saving the molecule with unclustered field point predictions
        xed_path_clustering: path to xed file saving the molecule with clustered field point predictions
        mol2_path_clustering: path to mol2 file saving the molecule with clustered field point predictions
        pymol_file_clustering: path to pymol file saving the molecule with clustered field point predictions
        cutoff: the threshold of cluster probability to make a prediciton
        dst_threshold: parameter for the clustering algorithm, determines cluster sizes
        sample_index: index of sample in data set to make predictions and visualizations on
    """
    (
        predicted_probs,
        predicted_fp,
        edges,
        atom_data,
        fieldpoint_data,
    ) = random_prediction(model, dataset, sample_index)
    # rescale, such that points will not be too large in visualization
    fieldpoint_data[:, 5] = fieldpoint_data[:, 5] / 10
    # write (unclustered) point cloud to file
    write_xed(
        edges,
        atom_data,
        fieldpoint_data,
        predicted_fp,
        predicted_probs,
        xed_path_cloud,
    )
    xed_to_mol2(xed_path_cloud, mol2_path_cloud)
    xed_mol2_to_pymol(mol2_path_cloud, xed_path_cloud, pymol_file_cloud)
    # define cluster model
    cluster_model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dst_threshold
    )
    # cluster the predicted field points
    cluster_classes = cluster_model.fit_predict(predicted_fp)
    cluster_centers, cluster_center_probs = merge_clusters(
        cluster_classes, predicted_fp, predicted_probs
    )
    cluster_centers = np.array(cluster_centers)
    cluster_center_probs = np.array(cluster_center_probs)
    write_cluster_xed(
        cluster_centers,
        cluster_center_probs,
        atom_data,
        edges,
        fieldpoint_data,
        cutoff,
        xed_path_clustering,
    )
    xed_to_mol2(xed_path_clustering, mol2_path_clustering)
    xed_mol2_to_pymol(mol2_path_clustering, xed_path_clustering, pymol_file_clustering)


def generate_pymol_samples(
    dataset, model, fp_type, cutoff, dst_threshold, load_path, number_samples=10
):
    """Create predictions and pymol visualizations for samples from dataset

    Args
        dataset: the data set to get samples 
        model: the (trained) SE(3)-Transformer model
        fp_type: field point type
        cutoff: the threshold of cluster probability to make a prediciton
        dst_threshold: parameter for the clustering algorithm, determines cluster sizes
        load_path: the checkpoint location from which the model was loaded
        number_samples: number of samples to make predictions and pymol visualizations of
    """
    if fp_type == -5:
        fp_dir = "negElec"
    if fp_type == -6:
        fp_dir = "posElec"
    if fp_type == -7:
        fp_dir = "vdw"
    if fp_type == -8:
        fp_dir = "hydrophobic"
    # generate 4 directories, one for each field point type
    path1 = f"visualization/pymol/pymolSamples/{fp_dir}"
    path2 = f"visualization/pymol/pymolSamples/{fp_dir}/data"
    path3 = f"visualization/pymol/pymolSamples/{fp_dir}/unclustered"
    path4 = f"visualization/pymol/pymolSamples/{fp_dir}/clustered"
    paths = [path1, path2, path3, path4]
    for path in paths:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
    # write info file:
    with open(path1 + "/info", "w") as f:
        f.write(f"Based on checkpoint path: {load_path}")
        f.write(f"threshold: {dst_threshold}")
    # random inds
    random.seed(3)
    randomInds = random.sample(range(1, len(dataset)), number_samples)
    # write
    for ind in range(1, number_samples):
        xedPathCloud = f"visualization/pymol/pymolSamples/{fp_dir}/data/cloud{ind}.xed"
        mol2PathCloud = (
            f"visualization/pymol/pymolSamples/{fp_dir}/data/cloud{ind}.mol2"
        )
        pymolFileCloud = (
            f"visualization/pymol/pymolSamples/{fp_dir}/unclustered/cloud{ind}.pml"
        )
        xedPathClustering = (
            f"visualization/pymol/pymolSamples/{fp_dir}/data/cluster{ind}.xed"
        )
        mol2PathClustering = (
            f"visualization/pymol/pymolSamples/{fp_dir}/data/cluster{ind}.mol2"
        )
        pymolFileClustering = (
            f"visualization/pymol/pymolSamples/{fp_dir}/clustered/cluster{ind}.pml"
        )
        write_ranomd_pymols(
            dataset,
            model,
            xedPathCloud,
            mol2PathCloud,
            pymolFileCloud,
            xedPathClustering,
            mol2PathClustering,
            pymolFileClustering,
            cutoff,
            dst_threshold,
            randomInds[ind],
        )


def xed_to_mol2(xed_file, mol2_file):
    """Convert xed to mol2 file

    Args:
        xed_file: path to xed file
        mol2_file: path to save mol2 file
    """
    element_dict = {
        "1": "H",
        "5": "B",
        "6": "C",
        "7": "N",
        "8": "O",
        "9": "F",
        "15": "P",
        "16": "S",
        "17": "Cl",
        "35": "Br",
        "53": "I",
        "14": "Si",
    }
    filein = xed_file.rsplit(".")[0]
    fi = open(xed_file)
    line = fi.readline()
    line_list = line.strip().split()
    num_bonds = int(line_list[2])
    num_atoms = int(line_list[1])
    # bonds
    num_lines = int(math.ceil(num_bonds / 5))
    num_last = num_bonds % 5
    if num_last == 0:
        num_last = 5
    fi.readline()
    bonds = []
    for i in range(num_lines):
        line2 = fi.readline()
        line_list = line2.strip().split()
        for j in range(0, len(line_list), 2):
            bonds.append([int(line_list[j]), int(line_list[j + 1])])
    print(bonds)
    # atoms
    atoms_lines = []
    for i in range(num_atoms):
        line2 = fi.readline()
        line_list = line2.strip().split()
        if line_list[0] != "-7" and line_list[0] != "-6":
            name = element_dict[line_list[0]]
            val = 0.0
            x, y, z = float(line_list[1]), float(line_list[2]), float(line_list[3])
            atoms_lines.append(
                f"%5d %-3s % 8.3f % 8.3f % 8.3f %-5s 1 TEST % 6.3f\n"
                % (i + 1, name, x, y, z, name, val)
            )
    # write to file:
    fo = open(mol2_file, "w")
    fo.write("@<TRIPOS>MOLECULE\n")
    fo.write("%s\n" % filein)
    fo.write("%d %d 1 0 0\n" % (len(atoms_lines), num_bonds))
    fo.write("SMALL\n")
    fo.write("NO_CHARGES\n")
    fo.write("\n")
    fo.write("@<TRIPOS>ATOM\n")
    # write atoms
    for atom_line in atoms_lines:
        fo.write(atom_line)
    # write bonds
    fo.write("@<TRIPOS>BOND\n")
    for ci, i in enumerate(bonds):
        fo.write("%5d %5d %5d 1\n" % (ci + 1, i[0], i[1]))
    fo.write("@<TRIPOS>SUBSTRUCTURE\n")
    fo.write("1  TEST  1 PERM 0 **** **** 0 ROOT\n")
    fo.close()
    fi.close()


def xed_mol2_to_pymol(
    mol2_file_path,
    xed_file,
    pymol_file="clusteringVisualization.pml",
    prediction_size_equal=False,
):
    """Convert an xed and mol2 file to pymol file

    Args
        mol2_file_path: path to mol2 file
        xed_file: path to xed file
        pymol_file: path to save pymol file 
        prediction_size_equal: if true, prediction spheres have a fixed size
    """
    pymol_lines = []
    sphere_scale_lines = []
    with open(xed_file) as file:
        num_fp = 0  # number true field points
        true_fp = []
        num_pred_fp = 0  # number predicted field points
        pred_fp = []
        for line in file:
            line_list = line.strip().split()
            if len(line_list) == 0:
                continue
            # original fieldpoints
            if line_list[0] == "-7":
                num_fp += 1
                fp_name = "orig" + str(num_fp)
                true_fp.append(fp_name)
                pseudo_atom_line = f"pseudoatom {fp_name}, pos=[{str(line_list[1])}, {str(line_list[2])} , {str(line_list[3])}]"
                pymol_lines.append(pseudo_atom_line)
                sphere_scale_lines.append(
                    f"set sphere_scale, {line_list[-1]}, {fp_name}"
                )
            # predicted fieldpoints
            if line_list[0] == "-6":
                num_pred_fp += 1
                fp_name = "fp" + str(num_pred_fp)
                pred_fp.append(fp_name)
                pseudo_atom_line = f"pseudoatom {fp_name}, pos=[{str(line_list[1])}, {str(line_list[2])} , {str(line_list[3])}]"
                pymol_lines.append(pseudo_atom_line)
                if prediction_size_equal:
                    sphere_size = 0.2
                else:
                    sphere_size = line_list[-1]
                    sphere_size_rescaled = (
                        np.clip(np.tanh(float(sphere_size) * 1000), 0.5, 1) / 10
                    )
                sphere_scale_lines.append(
                    f"set sphere_scale, {sphere_size_rescaled}, {fp_name}"
                )
        # spehere colour
        colour_cmd1 = "set sphere_color, yellow, (" + ",".join(true_fp) + ")"
        pymol_lines.append(colour_cmd1)
        colour_cmd2 = "set sphere_color, red, (" + ",".join(pred_fp) + ")"
        pymol_lines.append(colour_cmd2)
        # set sphere transparency
        pymol_lines.append("set sphere_transparency=0.6, (" + ",".join(true_fp) + ")")
        # load molecule
        mol2_write_path = "/".join(mol2_file_path.split("/")[4:])
        pymol_lines.append(f"load {mol2_write_path}, molecule")
        # make nice molecule colouring:
        pymol_lines.append('util.cba(144, "molecule", _self=cmd)')
        pymol_lines.append("set_bond stick_radius,0.1,molecule")
        # hide hydro
        pymol_lines.append("hide (hydro)")
        # write to file
        with open(pymol_file, "w") as f:
            for line in pymol_lines:
                f.write(line + "\n")
            for line in sphere_scale_lines:
                f.write(line + "\n")
            f.write("hide wire, (all)\n")
            f.write("show spheres, (" + ",".join(true_fp) + ")\n")
            f.write("show spheres, (" + ",".join(pred_fp) + ")\n")
