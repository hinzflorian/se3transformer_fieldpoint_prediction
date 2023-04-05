import logging
import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn
from dllogger import flush
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

##############################
from data_loading.molecule_data import FieldPointData, collate
from evaluation.evaluate import evaluate
from model import SE3TransformerMol
from model.fiber import Fiber
from runtime.arguments import PARSER
from runtime.callbacks import MoleculeLRSchedulerCallback
from runtime.clustering import *
from runtime.loggers import DLLogger, Logger, LoggerCollection
from runtime.utils import get_local_rank, init_distributed, using_tensor_cores

##############################


def load_model(model: nn.Module, path: pathlib.Path):
    """Load model from path

    Args:
        model (nn.Module): the SE(3)-Transformer model
        path (pathlib.Path): path to checkpoint
    """
    checkpoint = torch.load(str(path), map_location="cuda:" + str(args.device))
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])


def evaluation(
    model: nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    logger: Logger,
    args,
):
    """evaluation of model on training and test set

    Args:
        model (nn.Module): the SE(3)-Transformer model
        train_dataloader (DataLoader): data loader for training data
        test_dataloader (DataLoader): data loader for test data
        logger (Logger): the logger to save information
        args (_type_): further model arguments
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
    filterLevel = 0.005
    dst_threshold = 1
    distanceList = [0.5, 1, 1.5, 2]
    evaluate(
        model,
        train_dataloader,
        args,
        0,
        filterLevel,
        "train " + f"filterLevel {filterLevel}" + f"dst_threshold {dst_threshold}",
        dst_threshold,
        distanceList,
        logger,
    )
    flush()
    evaluate(
        model,
        test_dataloader,
        args,
        0,
        filterLevel,
        "test " + f"filterLevel {filterLevel}" + f"dst_threshold {dst_threshold}",
        dst_threshold,
        distanceList,
        logger,
    )
    flush()


def print_parameters_count(model: nn.Module):
    """print number of learnable parameters in model

    Args:
        model (nn.Module): neural network model
    """
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params_trainable}")


if __name__ == "__main__":
    ##############################
    # Args
    ##############################
    args = PARSER.parse_args()
    args.name = "loss7"
    args.file_type = "evaluation"
    args.number_molecules = 100000
    args.min_number_atoms = 0
    args.max_number_atoms = 100
    # args.radius = None
    args.radius = 7
    args.device = 7
    args.sigma = 0.2
    args.data_dir = pathlib.Path("./data")
    args.log_dir = pathlib.Path("results")
    args.dllogger_name = "dllogger_results_eval_hydrophobic.json"
    args.save_ckpt_path = pathlib.Path("./checkpoints/ckp_eval_hydrophobic")
    args.load_ckpt_path = pathlib.Path("./checkpoints/ckp_hydrophobic")
    # -5: negative electrostatic, -6: positive electrostatic, -7: vdw, -8: hydrophobic
    args.fieldpoint_type = -8
    args.descriptors_to_use = [
        "partialCharge",
        "atomSize",
        "logP",
        "degrees1Hot",
        "atomHybTypes1Hot",
    ]
    args.epochs = 100
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
    args.numVectors = 3
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
    # zarrPath="/disk2/sim_project/smilesData/zarrData/Total.zarr"
    zarrPath = "./data/Total_30122021.zarr"
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
    )
    train_set = data.train_set
    test_set = data.test_set
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    train_loader_eval = DataLoader(
        train_set,
        batch_size=args.batch_size_eval,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        test_set,
        batch_size=args.batch_size_eval,
        shuffle=False,
        collate_fn=collate,
        num_workers=args.num_workers,
        drop_last=True,
    )
    atom_feature_size = len(train_set[0][0].ndata["node_feats"][1])
    NODE_FEATURE_DIM = atom_feature_size
    args.feature_dim = NODE_FEATURE_DIM
    #####################################################################
    model = SE3TransformerMol(
        fiber_in=Fiber({0: NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.numVectors, 1: args.numVectors}),
        fiber_edge=Fiber({0: 0}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args),
    )
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.learnable_params = num_params_trainable

    callbacks = [MoleculeLRSchedulerCallback(logger, epochs=args.epochs)]
    print_parameters_count(model)
    logger.log_hyperparams(vars(args))
    with torch.no_grad():
        evaluation(model, train_loader_eval, val_loader, logger, args)
    logging.info("Evaluation finished successfully")
