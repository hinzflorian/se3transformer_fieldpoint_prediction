import numpy as np
import torch


def merge_clusters(cluster_classes, predicted_fp, predicted_probs):
    """Calculate (weighted) cluster centers and cluster probabilities

    Args
        cluster_classes: determines to which cluster point belongs
        predicted_fp: predicted field points
        predicted_probs: predicted probabilities of field points

    Return:
        cluster_centers: centers of clusters
        cluster_center_prob: sums of probabilities of points contained in
             corresponding clusters
    """
    clusters = np.unique(cluster_classes)
    cluster_centers = []
    cluster_center_prob = []
    for _, cluster in enumerate(clusters):
        cluster_idx = cluster_classes == cluster
        cluster_points = predicted_fp[cluster_idx, :]
        cluster_probs = predicted_probs[cluster_idx]
        cluster_proportions = cluster_probs / np.sum(cluster_probs)
        cluster_centers.append(
            np.sum(cluster_proportions.reshape((-1, 1)) * cluster_points, 0)
        )
        cluster_center_prob.append(np.sum(cluster_probs))
    return cluster_centers, cluster_center_prob


def cluster_preds(predicted_fp, predicted_probs, cluster_model, filter_level=0.005):
    """Calculate (weighted) cluster centers and cluster probabilities

    Args:
        predicted_fp: predicted positions of field points
        predicted_probs:predicted probability weights for fieldpoints
        cluster_model: the cluster model to assign clusters to points
        filter_level: the threshold for probability to make a prediction

    Return:
        cluster_centers_valid: centers of clusters which have high enough probability
        cluster_center_probs_valid: probability sum of of points in cluster, for clusters
            which exceed 'filter_level' probability
        
    """
    # only keep larger probs:
    valid_preds = predicted_probs > 0.0001
    predicted_probs_valid = predicted_probs[valid_preds]
    predicted_fp_valid = predicted_fp[valid_preds]
    # AgglomerativeClustering only works for at least 2 points:
    if torch.sum(valid_preds) == 1:
        cluster_centers_valid = predicted_fp_valid
        cluster_probs_valid = predicted_probs_valid
        return cluster_centers_valid, cluster_probs_valid
    cluster_classes = cluster_model.fit_predict(predicted_fp_valid.cpu())
    cluster_centers, cluster_probs = merge_clusters(
        cluster_classes,
        np.array(predicted_fp_valid.cpu()),
        np.array(predicted_probs_valid.cpu()),
    )
    cluster_centers = np.array(cluster_centers)
    cluster_probs = np.array(cluster_probs)
    # only make predictions at points with at least 'filter_level' probability
    valid_clusters = cluster_probs > filter_level
    cluster_centers_valid = cluster_centers[valid_clusters]
    cluster_probs_valid = cluster_probs[valid_clusters]
    return cluster_centers_valid, cluster_probs_valid
