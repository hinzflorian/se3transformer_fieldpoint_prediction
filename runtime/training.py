#############################################################################################################
import logging
import pathlib
from typing import List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from apex.optimizers import FusedAdam, FusedLAMB

#############################################################################################################
from data_loading.molecule_data import FieldPointData, collate
from dllogger import flush
from evaluation.evaluate import evaluate
from loss.loss_functions import apply_loss
from model import SE3TransformerMol
from model.fiber import Fiber
from torch.nn.modules.loss import _Loss
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from runtime.arguments import PARSER
from runtime.callbacks import BaseCallback, MoleculeLRSchedulerCallback
from runtime.loggers import DLLogger, Logger, LoggerCollection
from runtime.utils import get_local_rank, init_distributed, to_cuda, using_tensor_cores


#############################################################################################################
# Function definitions
#############################################################################################################
def save_state(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    path: pathlib.Path,
    callbacks: List[BaseCallback],
):
    """ Saves model, optimizer and epoch states to path (only once per node) """
    if get_local_rank() == 0:
        state_dict = (
            model.module.state_dict()
            if isinstance(model, DistributedDataParallel)
            else model.state_dict()
        )
        checkpoint = {
            "state_dict": state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        }
        for callback in callbacks:
            callback.on_checkpoint_save(checkpoint)

        torch.save(checkpoint, str(path))
        logging.info(f"Saved checkpoint to {str(path)}")


def load_state(
    model: nn.Module,
    path: pathlib.Path,
    callbacks: List[BaseCallback],
    load_scheduler,
    args,
):
    """ Loads model, optimizer and epoch states from path """
    checkpoint = torch.load(str(path), map_location="cuda:" + str(args.device))

    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])
    if load_scheduler:
        for callback in callbacks:
            callback.on_checkpoint_load(checkpoint)

    logging.info(f"Loaded checkpoint from {str(path)}")
    return checkpoint["epoch"]


def train_epoch(
    model,
    train_dataloader,
    loss_fn,
    epoch_idx,
    grad_scaler,
    optimizer,
    local_rank,
    callbacks,
    args,
):
    losses = []
    for i, batch in tqdm(
        enumerate(train_dataloader),
        total=len(train_dataloader),
        unit="batch",
        desc=f"Epoch {epoch_idx}",
        disable=(args.silent or local_rank != 0),
    ):
        g, y, _ = batch
        g = to_cuda(g)
        y = to_cuda(y)

        for callback in callbacks:
            callback.on_batch_start()

        with torch.cuda.amp.autocast(enabled=args.amp):  # type: ignore
            pred = model(g, {"0": g.ndata["node_feats"]}, None)  # type: ignore
            pred_list = [val for _, val in pred.items()]
            loss = loss_fn(y, pred_list, g, args.sigma) / args.accumulate_grad_batches
        grad_scaler.scale(loss).backward()
        print("args.dllogger_name:", args.dllogger_name)
        # gradient accumulation
        if (i + 1) % args.accumulate_grad_batches == 0 or (i + 1) == len(
            train_dataloader
        ):
            if args.gradient_clip:
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)

            grad_scaler.step(optimizer)
            grad_scaler.update()
            model.zero_grad(set_to_none=True)

        losses.append(loss.item())

    return np.mean(losses)


def train(
    model: nn.Module,
    loss_fn: _Loss,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    callbacks: List[BaseCallback],
    logger: Logger,
    args,
):
    """Train SE(3)-Transformer model

    Args:
        model (nn.Module): the SE(3)-Transformer model
        loss_fn (_Loss): the loss function used for training
        train_dataloader (DataLoader): data loader of training data
        test_dataloader (DataLoader): data loader of training data
        callbacks (List[BaseCallback]): 
        logger (Logger): 
        args (_type_): 
    """
    torch.cuda.set_device(args.device)
    device = torch.cuda.current_device()
    model.to(device=device)
    local_rank = get_local_rank()
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if dist.is_initialized():
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank
        )
        model._set_static_graph()
    model.train()
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    if args.optimizer == "adam":
        optimizer = FusedAdam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "lamb":
        optimizer = FusedLAMB(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.momentum, 0.999),
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

    for callback in callbacks:
        callback.on_fit_start(optimizer, args)

    load_scheduler = False
    epoch_start = (
        load_state(model, args.load_ckpt_path, callbacks, load_scheduler, args)
        if args.load_ckpt_path
        else 0
    )
    epoch_start = 0
    filter_level = 0.005
    dst_threshold = 1
    for epoch_idx in range(epoch_start, args.epochs):
        # write to log files:
        flush()
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch_idx)

        loss = train_epoch(
            model,
            train_dataloader,
            loss_fn,
            epoch_idx,
            grad_scaler,
            optimizer,
            local_rank,
            callbacks,
            args,
        )
        if dist.is_initialized():
            loss = torch.tensor(loss, dtype=torch.float, device=device)  # type: ignore
            torch.distributed.all_reduce(loss)
            loss = (loss / world_size).item()

        logging.info(f"Train loss: {args.accumulate_grad_batches*loss}")
        logger.log_metrics(
            {"train loss": args.accumulate_grad_batches * loss}, epoch_idx
        )

        for callback in callbacks:
            callback.on_epoch_end()

        if (
            args.save_ckpt_path is not None
            and args.ckpt_interval > 0
            and (epoch_idx + 1) % args.ckpt_interval == 0
        ):
            save_state(
                model,
                optimizer,
                epoch_idx,
                str(args.save_ckpt_path) + "_" + str(epoch_idx),
                callbacks,
            )
        if (epoch_idx + 1) % args.eval_interval == 0:
            distance_list = [0.5, 1, 1.5, 2]
            evaluate(
                model,
                train_loader_eval,
                args,
                0,
                filter_level,
                "train "
                + f"filter_level {filter_level}"
                + f"dst_threshold {dst_threshold}",
                dst_threshold,
                distance_list,
                logger,
            )
            evaluate(
                model,
                test_dataloader,
                args,
                0,
                filter_level,
                "test "
                + f"filter_level {filter_level}"
                + f"dst_threshold {dst_threshold}",
                dst_threshold,
                distance_list,
                logger,
            )


def print_parameters_count(model):
    """print number of learnable parameters in model

    Args:
        model (_type_): neural network model
    """
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Number of trainable parameters: {num_params_trainable}")


#############################################################################################################
if __name__ == "__main__":
    ############################################################################################
    # Args
    ############################################################################################
    args = PARSER.parse_args()
    args.name = "loss7"
    args.number_molecules = 100000  # number of molecules to load
    args.min_number_atoms = 0  # only load data with at least min_number_atoms atoms
    args.max_number_atoms = 100  # only load data with at most max_number_atoms atoms
    # construct graph based on putting edge beween nodes (atoms),
    # if they are not further apart than 'radius'.
    # If radius=None, use covalent bond graph topology
    args.radius = 7
    # alternatively
    # args.radius = 5
    args.device = 1  # GPU to use
    args.sigma = 0.2  # std for Gaussian mixtures
    args.log_dir = pathlib.Path("results")  # directory for logging
    args.dllogger_name = "dllogger_neg2.json"  # file to log
    args.save_ckpt_path = pathlib.Path(
        "./checkpoints/ckp_neg2"
    )  # file to save checkpoint
    args.load_ckpt_path = pathlib.Path("./checkpoints/ckp_neg")  # loading checkpoint
    # args.load_ckpt_path = None
    # fieldpoint type to load (-5: electrostatic negative, -6: electrostatic positive
    # , -7: vdw, -8: hydrophobic):
    args.fieldpoint_type = -5
    # which descriptors shall be used for training:
    args.descriptors_to_use = [
        "partialCharge",
        "atomSize",
        "logP",
        "degrees1Hot",
        "atomHybTypes1Hot",
    ]
    args.optimizer = "adam"  # optimizer to use
    args.learning_rate = 0.001  # initial learning rate
    args.min_learning_rate = 0.0002  # minimal learning rate
    args.momentum = 0.9
    args.weight_decay = 0.1
    args.epochs = 100
    args.batch_size = 50
    args.batch_size_eval = 10  # batch size for evaluation
    args.seed = None
    args.num_workers = 10
    args.amp = True
    args.gradient_clip = None
    args.accumulate_grad_batches = (
        10  # number of gradients of batches to accumulate before weight update
    )
    args.ckpt_interval = 1
    args.eval_interval = 100
    args.silent = False
    args.wandb = False
    args.benchmark = False
    args.task = "homo"
    args.precompute_bases = True
    args.num_layers = 7
    args.num_heads = 8
    args.channels_div = 2
    args.pooling = None
    args.norm = False
    args.use_layer_norm = True
    args.low_memory = False
    args.num_degrees = 3  # rotation orders to use
    args.num_channels = 32
    args.num_vectors = 3  # number of vectors to predict per atom
    args.percent_training = 0.8  # percent of data assigning as training
    args.number_conformations = (
        5  # maximal number of conformations to load per molecule
    )
    ############################################################################################
    # Loading Data
    ############################################################################################
    zarrPath = "data/Total_30122021.zarr"  # data location
    data = FieldPointData(
        args.descriptors_to_use,
        zarrPath,
        args.number_molecules,
        args.number_conformations,
        args.fieldpoint_type,
        args.radius,
        args.percent_training,
        args.min_number_atoms,
        args.max_number_atoms,
    )
    # data loader for training data:
    train_loader = DataLoader(
        data.train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    # for faster evaluation on training data:
    train_loader_eval = DataLoader(
        data.train_set,
        batch_size=args.batch_size_eval,
        shuffle=True,
        collate_fn=collate,
        pin_memory=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    # data loader for test data:
    test_loader = DataLoader(
        data.test_set,
        batch_size=args.batch_size_eval,
        shuffle=False,
        collate_fn=collate,
        num_workers=20,
        drop_last=True,
    )
    ############################################################################################
    # Defining SE(3) model
    ############################################################################################
    atom_feature_size = len(data.train_set[0][0].ndata["node_feats"][1])
    NODE_FEATURE_DIM = atom_feature_size
    args.featureDim = NODE_FEATURE_DIM
    model = SE3TransformerMol(
        fiber_in=Fiber({0: NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.num_vectors, 1: args.num_vectors}),
        fiber_edge=Fiber({0: 0}),
        output_dim=1,
        # use Tensor Cores more effectively
        tensor_cores=using_tensor_cores(args.amp),
        **vars(args),
    )
    ############################################################################################
    # Defining loss function
    ############################################################################################
    loss_fn = apply_loss
    ############################################################################################
    # Logging
    ############################################################################################
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
    logger.log_hyperparams(vars(args))
    callbacks = [MoleculeLRSchedulerCallback(logger, epochs=args.epochs)]
    ############################################################################################
    # Print model parameters
    ############################################################################################
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    args.learnable_params = num_params_trainable
    print_parameters_count(model)
    ############################################################################################
    # Start training
    ############################################################################################
    train(model, loss_fn, train_loader, test_loader, callbacks, logger, args)
    logging.info("Training finished successfully")

# Fieldpoints:
# blue: negative electrostatic field F- -5
# red: positive electrostatic field F+ -6
# yellow: Van der Vaals FO -7
# golden: hydrophopic FI -8
