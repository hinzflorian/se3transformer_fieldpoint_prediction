import logging

import torch
from loss.loss_functions import metric_batch
from runtime.utils import get_local_rank, to_cuda
from tqdm import tqdm


def evaluate(
    model,
    dataloader,
    args,
    epoch_nr,
    filter_level,
    train_or_test,
    dst_threshold,
    distance_list,
    logger,
):
    """evaluation of trained model

    Args
        model: the SE(3)-Transformer model
        dataloader: data loader for data set
        args: further model arguments
        epoch_nr: number of the epoch (info for logging)
        filter_level: the threshold for probability to make a prediction
        train_or_test: is it train or test set evaluation (for loggin info)
        dst_threshold: parameter for clustering algorithm, deterimining cluster sizes
        distance_list: list of distances for which to calculate the metrics
        logger: logger to document results

    Return:
        weighted_sensitivity_avg: averaged weighted sensitivity
        precision_avg: averaged precision
    """
    model.eval()
    true_positive_rate = []
    weighted_senitivity = []
    precision = []
    for _, batch in tqdm(
        enumerate(dataloader),
        total=len(dataloader),
        unit="batch",
        desc=f"Evaluation",
        leave=False,
        disable=(args.silent or get_local_rank() != 0),
    ):
        print("args.dllogger_name:", args.dllogger_name)
        g, y, _ = batch
        g = to_cuda(g)
        y = to_cuda(y)
        with torch.cuda.amp.autocast(enabled=args.amp):
            pred = model(g, {"0": g.ndata["node_feats"]}, None)
            pred_list = [val.detach() for _, val in pred.items()]
            evaluation_metrics = metric_batch(
                pred_list, g, y, distance_list, filter_level, dst_threshold
            )
            true_positive_rate.append(evaluation_metrics[0])
            weighted_senitivity.append(evaluation_metrics[1])
            precision.append(evaluation_metrics[2])
    true_positive_rate = torch.stack(true_positive_rate)
    weighted_sensitivity = torch.stack(weighted_senitivity)
    precision = torch.stack(precision)
    true_positive_rate_avg = torch.mean(true_positive_rate, 0)
    weighted_sensitivity_avg = torch.mean(weighted_sensitivity, 0)
    precision_avg = torch.mean(precision, 0)
    # logging
    logging.info("Evaluation of metric on" + train_or_test)
    logging.info(f"true_positive_rate_avg: {true_positive_rate_avg}")
    logging.info(f"weighted_sensitivity_avg: {weighted_sensitivity_avg}")
    logging.info(f"precision_avg: {precision_avg}")
    logger.log_metrics({"Evaluation of metric on": train_or_test})
    logger.log_metrics(
        {"true_positive_rate_avg": [val.item() for val in true_positive_rate_avg]},
        epoch_nr,
    )
    logger.log_metrics(
        {"weighted_sensitivity_avg": [val.item() for val in weighted_sensitivity_avg]},
        epoch_nr,
    )
    logger.log_metrics(
        {"precision_avg": [val.item() for val in precision_avg]}, epoch_nr
    )
    return (
        weighted_sensitivity_avg,
        precision_avg,
    )
