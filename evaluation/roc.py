# The purpose of this script is to generate ROC curve data and save it to file
import logging
import os
import pathlib

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from data_loading.molecule_data import FieldPointData, collate
from dllogger import flush
from loss.loss_functions import apply_loss
from model import SE3TransformerMol
from model.fiber import Fiber
from runtime.arguments import PARSER
from runtime.loggers import DLLogger, LoggerCollection
from runtime.utils import get_local_rank, init_distributed, using_tensor_cores
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

##############################
# Own libraries
##############################
from evaluation.evaluate import evaluate


##############################
def load_model(model: nn.Module, path: pathlib.Path):
    """Load model from path

    Args
        model: the SE(3)-Transformer model

    Return:
    """
    checkpoint = torch.load(str(path), map_location="cuda:" + str(args.device))
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])


def calculate_sensitivity_precision(
    model: nn.Module, val_dataloader: DataLoader, args,
):
    """Calculate the sensitivity and precision for different filter levels

    Args:
        model (nn.Module): the (trained) SE(3)-Transformer model 
        val_dataloader (DataLoader): validation data loader
        args (_type_): _description_
    """
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        model._set_static_graph()
    model.eval()
    load_model(model, args.load_ckpt_path)
    filter_levels = np.linspace(0.1, 0.001, num=50)
    # we want to calculate senstivity and precision for distance 0.5:
    distance_list = [0.5]
    # save sensitivity and precision in this list
    sen_prec = []
    for ind, filterLevel in enumerate(filter_levels):
        print(f"filterLevel {filterLevel}, number {ind} of {len(filter_levels)}")
        (weighted_sensitivity, precision,) = evaluate(
            model,
            val_dataloader,
            args,
            0,
            filterLevel,
            "test "
            + f"filterLevel {filterLevel}"
            + f"dst_threshold {args.dst_threshold}",
            args.dst_threshold,
            distance_list,
            logger,
        )
        sen_prec.append([weighted_sensitivity[0], precision[0]])
        flush()
    sen_prec_array = np.array(torch.tensor(sen_prec))
    sensitivity = sen_prec_array[:, 0]
    precision = sen_prec_array[:, 1]
    # write sensitivity and precision to file:
    file_dir = os.path.dirname(os.path.realpath(__file__))
    image_path = "/images/shortVersion"
    np.savetxt(
        file_dir
        + image_path
        + f"fieldpoint_{np.abs(args.fieldpoint_type)}_r1_sensitivity",
        sensitivity,
        delimiter=",",
    )
    np.savetxt(
        file_dir
        + image_path
        + f"fieldpoint_{np.abs(args.fieldpoint_type)}__r1_filter_levels",
        filter_levels,
        delimiter=",",
    )
    np.savetxt(
        file_dir
        + image_path
        + f"fieldpoint_{np.abs(args.fieldpoint_type)}_r1_precision",
        precision,
        delimiter=",",
    )


if __name__ == "__main__":
    ##############################
    # Args
    ##############################
    args = PARSER.parse_args()
    args.name = "loss7"
    args.fileType = "evaluation"
    args.number_molecules = 100000
    args.min_number_atoms = 0
    args.max_number_atoms = 100
    args.radius = 7
    # args.radius = None #that means using covalent bond topology
    # alternatively:
    # args.radius=5
    args.device = 2
    args.sigma = 0.2
    args.data_dir = pathlib.Path("./data")
    args.log_dir = pathlib.Path("results")
    args.dllogger_name = "dllogger_results_eval_hydrophobic.json"
    args.save_ckpt_path = pathlib.Path("./checkpoints/ckp_eval_hydrophobic")
    args.load_ckpt_path = pathlib.Path("./checkpoints/ckp_hydrophobic")
    args.fieldpoint_type = -8
    args.descriptors_to_use = [
        "partialCharge",
        "atomSize",
        "logP",
        "degrees1Hot",
        "atomHybTypes1Hot",
    ]
    args.dst_threshold = 1
    args.generateImages = True
    args.optimizer = "adam"
    args.learning_rate = 0.002
    args.min_learning_rate = 0.0003
    args.momentum = 0.9
    args.weight_decay = 0.1
    args.epochs = 250
    args.batch_size = 100
    args.batch_size_eval = 300
    args.seed = None
    args.num_workers = 0
    args.amp = True
    args.gradient_clip = None
    args.accumulate_grad_batches = 500
    args.ckpt_interval = 1
    args.eval_interval = 40
    args.silent = False
    args.benchmark = False
    args.task = "homo"
    args.precompute_bases = True
    args.num_layers = 7
    args.num_heads = 8
    args.channels_div = 2
    args.pooling = None
    args.norm = False  # True is bad
    args.use_layer_norm = True
    args.low_memory = False
    args.num_degrees = 3
    args.num_channels = 32
    args.num_vectors = 3
    args.percent_training = 0.8
    ##############################
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    logging.getLogger().setLevel(
        logging.CRITICAL if local_rank != 0 or args.silent else logging.INFO
    )
    logging.info("====== SE(3)-Transformer ======")
    logging.info("|      Training procedure     |")
    logging.info("===============================")
    loggers = [DLLogger(save_dir=args.log_dir, filename=args.dllogger_name)]
    logger = LoggerCollection(loggers)
    #####################################################################
    zarrPath = "/home/florian/Data/Total_30122021.zarr"
    number_conformations = 5
    data = FieldPointData(
        args.descriptors_to_use,
        zarrPath,
        args.number_molecules,
        number_conformations,
        args.fieldpoint_type,
        args.radius,
        args.percent_training,
        args.min_number_atoms,
        args.max_number_atoms,
        lengthTest=1000,
    )
    train_set = data.train_set
    test_set = data.test_set
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size_eval,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=True,
    )
    atom_feature_size = len(train_set[0][0].ndata["node_feats"][1])
    NODE_FEATURE_DIM = atom_feature_size
    args.featureDim = NODE_FEATURE_DIM
    #####################################################################
    model = SE3TransformerMol(
        fiber_in=Fiber({0: NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_vectors, 1: args.num_vectors}),
        fiber_edge=Fiber({0: 0}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args),
    )
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.learnable_params = num_params_trainable

    loss_fn = apply_loss
    logger.log_hyperparams(vars(args))
    with torch.no_grad():
        calculate_sensitivity_precision(model, test_loader, args)
    logging.info("Evaluation finished successfully")
