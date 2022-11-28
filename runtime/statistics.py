import torch
from data_loading.molecule_data import FieldPointData

from runtime.arguments import PARSER

if __name__ == "__main__":
    ###################################################################################################
    # Args
    ###################################################################################################
    args = PARSER.parse_args()
    args.number_molecules = 100000
    args.nr_atoms_min = 0
    args.nr_atoms_max = 100
    # args.radius=None
    args.radius = 7
    args.fieldpointType = -7
    args.predictorsToUse = [
        "partialCharge",
        "atomSize",
        "logP",
        "degrees1Hot",
        "atomHybTypes1Hot",
    ]
    args.percent_training = 0.8
    ###################################################################################################
    zarrPath = "/home/florian/Data/Total_30122021.zarr"
    number_conformations = 5
    data = FieldPointData(
        args.predictorsToUse,
        zarrPath,
        args.number_molecules,
        number_conformations,
        args.fieldpoint_type,
        args.radius,
        args.percent_training,
        args.nr_atoms_min,
        args.nr_atoms_max,
    )
    train_set = data.train_set
    et = data.testSet
    max_degree = 0
    total_len = len(train_set)
    train_set[0]
    for ind in range(0, total_len):
        degrees = train_set[ind][2][0]
        local_max_degree = torch.max(degrees)
        if max_degree < local_max_degree:
            max_degree = local_max_degree
        if ind % 5000:
            print(max_degree)
            print("progress: ", 100 * ind / total_len, " %")
    print(max_degree)

# Fieldpoints:
# blue: negative electrostatic field F- -5
# red: positive electrostatic field F+ -6
# yellow: Van der Vaals FO -7
# golden: hydrophopic FI -8
