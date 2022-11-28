import pathlib

import torch
import torch.distributed as dist
import torch.nn as nn
##############################
from data_loading.molecule_data import FieldPointData
from model import SE3TransformerMol
from model.fiber import Fiber
from runtime.arguments import PARSER
from runtime.utils import get_local_rank, init_distributed, using_tensor_cores
from torch.nn.parallel import DistributedDataParallel
from visualization.visualization import generate_pymol_samples


##############################
def load_model(model: nn.Module, path: pathlib.Path):
    """Load model from path

    Args
        model: the SE(3)-Transformer model
    """
    checkpoint = torch.load(str(path), map_location="cuda:" + str(args.device))
    if isinstance(model, DistributedDataParallel):
        model.module.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint["state_dict"])


def pymol_generation(
    model: nn.Module, data, args,
):
    """Generation of of pymol visualizations

    Args
        model: the (trained) SE(3)-Transformer model 
        data: the data from which to randomly pick samples, make predictions,
                and generate pymol visualizations
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
    generate_pymol_samples(
        data,
        model,
        args.fieldpoint_type,
        args.cutoff,
        args.dst_threshold,
        args.load_ckpt_path,
        200,
    )


if __name__ == "__main__":
    ##############################
    # Args
    ##############################
    args = PARSER.parse_args()
    args.percent_training = 0.8  # percentage of data going to training
    args.number_molecules = 100000
    args.min_number_atoms = 0
    args.max_number_atoms = 100
    # alternatively:
    args.radius = 7
    # args.radius = None
    # args.radius=5
    args.device = 3
    args.sigma = 0.2
    # hydrophobic field points:
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
    args.dst_threshold = 1
    args.cutoff = 0.005
    args.batch_size = 50
    args.batch_size_eval = 10
    args.seed = None
    args.num_workers = 10
    args.amp = True
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
    args.num_degrees = 3
    args.num_channels = 32
    args.numVectors = 3
    ##############################
    is_distributed = init_distributed()
    local_rank = get_local_rank()
    #####################################################################
    # zarrPath="/disk2/sim_project/smilesData/zarrData/Total.zarr"
    zarrPath = "/home/florian/Data/Total_30122021.zarr"
    number_molecules = args.number_molecules
    number_conformations = 5
    typeFieldpoint = args.fieldpoint_type
    percent_training = args.percent_training
    data = FieldPointData(
        args.descriptors_to_use,
        zarrPath,
        number_molecules,
        number_conformations,
        typeFieldpoint,
        args.radius,
        percent_training,
        args.min_number_atoms,
        args.max_number_atoms,
    )
    train_set = data.train_set
    test_set = data.test_set
    #####################################################################
    atom_feature_size = len(train_set[0][0].ndata["node_feats"][1])
    NODE_FEATURE_DIM = atom_feature_size
    args.featureDim = NODE_FEATURE_DIM
    #####################################################################
    model = SE3TransformerMol(
        fiber_in=Fiber({0: NODE_FEATURE_DIM}),
        fiber_out=Fiber({0: args.numVectors, 1: args.numVectors}),
        fiber_edge=Fiber({0: 0}),
        output_dim=1,
        tensor_cores=using_tensor_cores(args.amp),  # use Tensor Cores more effectively
        **vars(args),
    )
    pymol_generation(model, test_set, args)
