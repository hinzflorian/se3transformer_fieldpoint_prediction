"""This module contains loss function building blocks"""

import runtime.clustering as clustering
import torch
from sklearn.cluster import AgglomerativeClustering


######################################################################
def kl_approx(f_centers, f_probs, g_centers, g_probs, sigma):
    """Calculate an approxmation of KL(f|g)

    Args
        f_centers: points of discrete distribution on f_centers 
        f_probs: corresponding probabilities for f_centers
        g_centers: centers of gaussian mixture g
        g_weights: corresponding probabilities for g_centers

    Return:
        kl_mc_estimator: approximation of KL(f|g)
    """
    # distence between centers:
    center_diff = torch.unsqueeze(f_centers, dim=1) - torch.unsqueeze(g_centers, dim=0)
    center_dist = torch.norm(center_diff, dim=2)
    g_val = torch.sum(
        torch.exp(-0.5 * sigma ** (-2) * center_dist ** 2) * g_probs, dim=1
    )
    # g_val cannot be smaller than 0, it just occurs by numerics
    g_val[g_val < 0] = 0
    eps = 1e-12
    log_f_div_g = torch.log(f_probs / (g_val + eps) + eps)
    kl_mc_estimator = torch.sum(f_probs * log_f_div_g)
    return kl_mc_estimator


def weighted_vector_length_sum(predicted_vectors, predicted_probs):
    """calculate the sum of vector lengths, scaled with corresponding probability

    Args
        predicted_vectors: vectors as predicted by model
        predicted_probs:  probabilities corresponding to vectors

    Return:
        scaled_sum_weighted_vector: rescaled and weighted (by probability) sum of vector lenghts
    """
    vector_lengths = torch.norm(predicted_vectors, dim=2).reshape((-1))
    weighted_vector_lengths = predicted_probs * vector_lengths
    scaled_sum_weighted_vector = 10 * torch.sum(weighted_vector_lengths)
    print("scaled_sum_weighted_vector", scaled_sum_weighted_vector)
    return scaled_sum_weighted_vector


def loss1(
    true_centers,
    true_charges,
    predicted_fp,
    predicted_weights,
    sigma,
    predicted_vectors,
):
    """Loss function consisting of symmetrized Kullback Leibler divergence, penalization of large
        probability weights, penalization of vector lengths

    Args
        true_centers: location of field points
        true_charges: charges of field points
        predicted_fp: location of predicted field point
        predicted_weights: predicted probability weights of field points
        sigma: standard deviation to use in Gaussian mixture
        predicted_vectors: predicted vectors

    Return:
        loss_total: sum of different losses (symmetrized Kullback Leibler divergence, penalization of large
        probability weights, penalization of vector lenghts)
    """
    predicted_probs = torch.nn.Softmax(dim=0)(predicted_weights)
    true_probs = torch.abs(true_charges) / torch.sum(torch.abs(true_charges))
    # kl loss
    kl_approx2 = kl_approx(
        true_centers, true_probs, predicted_fp, predicted_probs, sigma
    )
    kl_approx2_reverse = kl_approx(
        predicted_fp, predicted_probs, true_centers, true_probs, sigma
    )
    scaled_sum_weighted_vector = (
        weighted_vector_length_sum(predicted_vectors, predicted_probs) / 100
    )
    scaling_factor = torch.sum(torch.abs(true_charges))
    probs_loss = torch.sum(predicted_probs ** 2) * 10
    print("probs_loss:", probs_loss)
    loss_total = scaling_factor * (
        kl_approx2 + kl_approx2_reverse + probs_loss + scaled_sum_weighted_vector
    )
    return loss_total


def apply_loss(y, pred, g, sigma):
    """Calculate the loss function for a batch of conformations, predictions of fieldpoints

    Args
        y: list of fieldpoint data (each a nx4 tensor, n fieldpoints, 4 (3 coords and 1 fieldvalue)) 
        pred: 2 tuple: pred[0]: (weights nAtoms) x WeightsPerAtom x (weight scalar) tensor
                       pred[1]: (weights nAtoms) x WeightsPerAtom x (prediction vector) tensor
        g: Graph object (consisting of "batch size" unconnnected subgraphs)
        sigma: standard deviation to use in Gaussian mixture

    Return:
        average_loss: average loss (averaged over conformation graphs in batch)
    """
    num_nodes_per_graph = g.batch_num_nodes()
    last_node_index = 0
    loss = 0
    for graph_index, number_nodes in enumerate(num_nodes_per_graph):
        predicted_vectors = pred[1][last_node_index : last_node_index + number_nodes]
        predicted_centers = (
            torch.unsqueeze(g.ndata["x"], 1)[
                last_node_index : last_node_index + number_nodes
            ]
            + predicted_vectors
        ).view((-1, 3))
        predicted_weights = pred[0][
            last_node_index : last_node_index + number_nodes,
        ].reshape((-1))
        charges = torch.abs(y[graph_index][:, 3])
        true_centers = y[graph_index][:, :3]
        loss_new = loss1(
            true_centers,
            charges,
            predicted_centers,
            predicted_weights,
            sigma,
            predicted_vectors,
        )
        print("loss", loss_new)
        if torch.isnan(loss_new):
            print("Error: Nan value in loss")
        loss += loss_new
        last_node_index += number_nodes
    average_loss = 1 / len(num_nodes_per_graph) * loss
    print("average_loss:", average_loss)
    return average_loss


######################################################################
def metric_batch(pred, g, y, distance_list, filter_level, dst_threshold):
    """Calculate several metrics on different distance levels for one batch

    Args:
        pred: 2 tuple: pred[0]: (weights nAtoms) x WeightsPerAtom x (weight scalar) tensor
                       pred[1]: (weights nAtoms) x WeightsPerAtom x (prediction vector) tensor
        g: Graph object (consisting of "batch size" unconnnected subgraphs)
        y: list of fieldpoint data (each a nx4 tensor, n fieldpoints, 4 (3 coords and 1 fieldvalue)) 
        distance_list: list of distances for which to calculate the metrics
        filter_level: threshold of cluster probability to make a prediction
        dst_threshold: parameter determining the cluster sizes

    Return:
        true_positive_rate: true positive rate (#found fieldpoints)/(#total fieldpoints)
        weighted_sensitivity: (#sum charge found)/(#total sum charge)
        precision / len(num_nodes_per_graph):  average precision (tp/(tp+fp))
    """
    corr_device = pred[0].get_device()
    # for clustered predictions:
    cluster_model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=dst_threshold
    )
    true_positive_rate = torch.zeros(len(distance_list), device=corr_device)
    weighted_sensitivity = torch.zeros(len(distance_list), device=corr_device)
    precision = torch.zeros(len(distance_list), device=corr_device)
    for ind, distance in enumerate(distance_list):
        num_nodes_per_graph = g.batch_num_nodes()
        last_node_index = 0
        number_fieldpoints = 0
        sum_charges = 0
        for graph_index, number_nodes in enumerate(num_nodes_per_graph):
            predicted_vectors = pred[1][
                last_node_index : last_node_index + number_nodes
            ]
            predicted_centers = (
                torch.unsqueeze(g.ndata["x"], 1)[
                    last_node_index : last_node_index + number_nodes
                ]
                + predicted_vectors
            ).view((-1, 3))
            predicted_weights = pred[0][
                last_node_index : last_node_index + number_nodes,
            ].reshape((-1))
            last_node_index = last_node_index + number_nodes
            predicted_probs = torch.nn.Softmax(dim=0)(predicted_weights)
            charges = torch.abs(y[graph_index][:, 3])
            true_centers = y[graph_index][:, :3]
            diff_fp = true_centers.reshape((-1, 1, 3)) - predicted_centers.reshape(
                (1, -1, 3)
            )
            distFp = torch.norm(diff_fp, dim=2)
            fp_in_radius = distFp < distance
            weights_per_fp = torch.sum(predicted_probs * fp_in_radius, dim=1)
            # use prediction threshold:
            predict_fp = weights_per_fp > 0.01
            number_fieldpoints += len(predict_fp)
            # clustering results:
            cluster_centers, _ = clustering.cluster_preds(
                predicted_centers, predicted_probs, cluster_model, filter_level
            )
            cluster_centers = torch.tensor(cluster_centers, device=corr_device)
            diff_fp_clusters = true_centers.reshape(
                (-1, 1, 3)
            ) - cluster_centers.reshape((1, -1, 3))
            distFp_clusters = torch.norm(diff_fp_clusters, dim=2)
            fp_in_radius_clusters = distFp_clusters < distance
            foundFp_cluster = torch.any(fp_in_radius_clusters, dim=1)
            true_positive_rate[ind] += torch.sum(foundFp_cluster)
            # for weighted TPR
            weighted_sensitivity[ind] += torch.sum(charges[foundFp_cluster])
            sum_charges += torch.sum(charges)
            # precision
            fp_prediction_true_false = torch.any(fp_in_radius_clusters, dim=0)
            precision[ind] += torch.sum(fp_prediction_true_false) / len(
                fp_prediction_true_false
            )
        true_positive_rate[ind] = true_positive_rate[ind] / number_fieldpoints
        weighted_sensitivity[ind] = weighted_sensitivity[ind] / sum_charges
    # metric results:
    # true_positive_rate[ind]: (found fieldpoints)/(total fieldpoints)
    # weighted_sensitivity[ind]: (sum charge found)/(total sum charge)
    metrics = [
        true_positive_rate,
        weighted_sensitivity,
        precision / len(num_nodes_per_graph),
    ]
    return metrics
