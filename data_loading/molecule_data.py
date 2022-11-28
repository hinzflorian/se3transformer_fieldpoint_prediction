import dgl
import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from data_loading.construct_predictors import atom_size_from_num, rbf_expand

DTYPE = torch.float32


class FieldpointDataset(Dataset):
    def __init__(
        self,
        descriptors_to_use,
        data_set_raw,
        fieldpoint_type,
        centers_partial_charge,
        centers_atom_size,
        centerslogP,
        radius,
    ):
        """ class to acces the field point data
        Args:
            descriptors_to_use: the set of descriptors to use
            dataset_raw: dictionary ('moleculeInfo','atoms','fieldPoints','xeds','edges','paramsArray','paramsSplitInds','conformationInfo')
                            of numpy arrays
            fieldpoint_type: the type of fieldpoint to load
            centers_partial_charge: centers for radial basis function expansion
            centers_atom_size: centers for radial basis function expansion
            centerslogP: centers for radial basis function expansion
            radius: an edge will be contained in constructed graph if distance between two nodes is less than 'radius'
        """
        self.dataset_raw = data_set_raw
        self.fieldpoint_type = fieldpoint_type
        self.centers_partial_charge = centers_partial_charge
        self.centers_atom_size = centers_atom_size
        self.centerslogP = centerslogP
        self.radius = radius
        self.descriptors_to_use = descriptors_to_use

    def __len__(self):
        """Calculate the length of the data set

        Returns:
            data_set_len: length of data set
        """
        # calculate length of dataset
        data_set_len = len(self.dataset_raw["conformationInfo"]) - 1
        return data_set_len

    def get_item_data(self, idx):
        """Fetch data set sample for given index

        Args:
            idx: index of sample to return data for

        Returns:
            edges: edge information of the molecule sample
            atom_data: some descriptors associated to atom
            xeds: the xed charge
            fieldpoint_data:
            parameters:
        """
        idxStartEdge = self.dataset_raw["conformationInfo"][idx][0]
        idxEndEdge = self.dataset_raw["conformationInfo"][idx + 1][0]
        edges = self.dataset_raw["edges"][:, idxStartEdge:idxEndEdge]
        # atom data
        idxStartConf = self.dataset_raw["conformationInfo"][idx][1]
        idxEndConf = self.dataset_raw["conformationInfo"][idx + 1][1]
        atom_data = self.dataset_raw["atoms"][idxStartConf:idxEndConf]
        # fieldpoint data
        idxStartFp = self.dataset_raw["conformationInfo"][idx][-1]
        idxEndFp = self.dataset_raw["conformationInfo"][idx + 1][-1]
        fieldpointsAll = self.dataset_raw["fieldPoints"][idxStartFp:idxEndFp, :]
        fieldpoint_data = fieldpointsAll[
            fieldpointsAll[:, 0] == self.fieldpoint_type, :
        ]
        # start index
        idxStartPrms = self.dataset_raw["paramsSplitInds"][idx]
        idxEndPrms = self.dataset_raw["paramsSplitInds"][idx + 1]
        parameters = self.dataset_raw["paramsArray"][idxStartPrms:idxEndPrms]
        # xeds
        idxStartXed = self.dataset_raw["conformationInfo"][idx][2]
        idxEndXed = self.dataset_raw["conformationInfo"][idx + 1][2]
        xeds = self.dataset_raw["xeds"][idxStartXed:idxEndXed]
        return edges, atom_data, xeds, fieldpoint_data, parameters

    def __getitem__(self, idx):
        idxStartConf = self.dataset_raw["conformationInfo"][idx][1]
        idxEndConf = self.dataset_raw["conformationInfo"][idx + 1][1]
        atom_data = self.dataset_raw["atoms"][idxStartConf:idxEndConf]
        positions = atom_data[:, 1:4]
        # atom one hot
        atomHybTypes1Hot = self.dataset_raw["atomHybTypes1Hot"][idxStartConf:idxEndConf]
        # for edges
        idxStartEdge = self.dataset_raw["conformationInfo"][idx][0]
        idxEndEdge = self.dataset_raw["conformationInfo"][idx + 1][0]
        edges = self.dataset_raw["edges"][:, idxStartEdge:idxEndEdge]
        src = edges[0]
        dst = edges[1]
        G_original = dgl.graph((src, dst))
        degrees = (G_original.in_degrees() + G_original.out_degrees()).reshape((-1, 1))
        degrees1Hot = torch.eye(4, 4)[degrees - 1].reshape((-1, 4))
        # alternative edges
        # radius=2.5
        if self.radius is not None:
            edges2 = edgeConstruction(positions, edges, self.radius)
            edges = edges2
        src = edges[0]
        dst = edges[1]
        G = dgl.graph((src, dst))
        # atom postisitions
        G.ndata["x"] = torch.tensor(positions, dtype=DTYPE)  # [num_atoms,3]
        # partial_charge
        partial_charge = atom_data[:, -3]
        partial_charge_expanded = rbf_expand(
            partial_charge, self.centers_partial_charge
        )
        # atom size
        atomNums = atom_data[:, 0].astype(int)
        atomSizes = atom_size_from_num(atomNums)
        atom_size_expanded = rbf_expand(atomSizes, self.centers_atom_size)
        # logP
        logP = atom_data[:, -2]
        logpExpanded = rbf_expand(logP, self.centerslogP)
        predictorsDict = {
            "partialCharge": partial_charge_expanded,
            "atomSize": atom_size_expanded,
            "logP": logpExpanded,
            "degrees1Hot": degrees1Hot,
            "atomHybTypes1Hot": atomHybTypes1Hot,
        }
        predictorList = [
            predictorsDict[predictorName] for predictorName in self.descriptors_to_use
        ]

        features = np.hstack(predictorList)
        G.ndata["node_feats"] = torch.tensor(np.expand_dims(features, -1), dtype=DTYPE)
        src_coord = positions[src, :]
        dst_coord = positions[dst, :]
        G.edata["rel_pos"] = torch.tensor(dst_coord - src_coord, dtype=DTYPE)
        # fieldpoints coordinates (target variable)
        idxStartFp = self.dataset_raw["conformationInfo"][idx][-1]
        idxEndFp = self.dataset_raw["conformationInfo"][idx + 1][-1]
        fieldpointsAll = self.dataset_raw["fieldPoints"][idxStartFp:idxEndFp, :]
        fieldpoint_data = fieldpointsAll[
            fieldpointsAll[:, 0] == self.fieldpoint_type, :
        ]
        # only keep position and charge
        fieldpoint_data = torch.tensor(fieldpoint_data[:, [1, 2, 3, 5]])
        if len(fieldpoint_data) == 0:
            print("No Fieldpoint")
        idxStartPrms = self.dataset_raw["paramsSplitInds"][idx]
        idxEndPrms = self.dataset_raw["paramsSplitInds"][idx + 1]
        parameters = torch.tensor(
            self.dataset_raw["paramsArray"][idxStartPrms:idxEndPrms]
        )
        # xeds
        idxStartXed = self.dataset_raw["conformationInfo"][idx][2]
        idxEndXed = self.dataset_raw["conformationInfo"][idx + 1][2]
        partial_charges_xed = torch.tensor(
            self.dataset_raw["xeds"][idxStartXed:idxEndXed, 5]
        )
        xed_coords = torch.tensor(self.dataset_raw["xeds"][idxStartXed:idxEndXed, 1:4])
        #################
        partial_charges_atoms = torch.tensor(atom_data[:, 5])
        return (
            G,
            fieldpoint_data,
            (
                degrees,
                partial_charges_atoms,
                partial_charges_xed,
                xed_coords,
                parameters,
            ),
        )


class FieldPointData:
    """field point data set"""

    def __init__(
        self,
        descriptors_to_use,
        zarrPath,
        number_molecules,
        number_conformations,
        fieldpoint_type,
        radius,
        percentTraining,
        min_number_atoms,
        max_number_atoms,
        lengthTest=None,
    ):
        """
        Args:
                descriptors_to_use: set of descriptors to use
                zarrPath: path to the zarr directory to read the data from 
                number_molecules: number of molecules to load
                number_conformations: number of conformations to load per molecule
                fieldpoint_type: the type of fieldpoint to load
                radius: an edge will be contained in constructed graph if distance between two nodes is less than 'radius'
                percentTraining: percentage of molecules to put into training set
                min_number_atoms: only load molecules with at least min_number_atoms atoms
                max_number_atoms: only load molecules with at most max_number_atoms atoms
                lengthTest (optional): If given, test test set will have this length (otherwise total-train_length)
        """
        self.descriptors_to_use = descriptors_to_use
        self.data_paths = zarrPath
        self.number_molecules = number_molecules
        self.number_conformations = number_conformations
        self.fieldpoint_type = fieldpoint_type
        maxNrMolecules = num_molecules(zarrPath)
        np.random.seed(42)
        permutedIndices = np.random.permutation(
            np.minimum(maxNrMolecules, self.number_molecules)
        )
        trainIndices = np.sort(
            permutedIndices[: int(len(permutedIndices) * percentTraining)]
        )
        if lengthTest == None:
            testIndices = np.sort(
                permutedIndices[int(len(permutedIndices) * percentTraining) :]
            )
        else:
            testIndices = np.sort(
                permutedIndices[
                    int(len(permutedIndices) * percentTraining) : int(
                        len(permutedIndices) * percentTraining
                    )
                    + lengthTest
                ]
            )

        trainIndicesRestricted = self.filter_molecule_indices(
            zarrPath, trainIndices, min_number_atoms, max_number_atoms
        )
        testIndicesRestricted = self.filter_molecule_indices(
            zarrPath, testIndices, min_number_atoms, max_number_atoms
        )
        self.dataset_raw_train = self.load_molecule_data(
            self.data_paths, trainIndicesRestricted
        )
        self.datasetRawTest = self.load_molecule_data(
            self.data_paths, testIndicesRestricted
        )
        minCharge = -1.141
        maxCharge = 1.703
        self.centers_partial_charge = np.linspace(minCharge, maxCharge, 15)
        # support points for radial basis expansion of logP
        minlogP = -1.95
        maxlogP = 0.8857
        self.centerslogP = np.linspace(minlogP, maxlogP, 20)
        # support points for radial basis expansion of atom size
        atomNums = self.dataset_raw_train["atoms"][:, 0].astype(int)
        atomSizes = atom_size_from_num(atomNums)
        minAtomSize = np.min(atomSizes)
        maxAtomSize = np.max(atomSizes)
        self.centers_atom_size = np.linspace(minAtomSize, maxAtomSize, 10)
        # define training set
        self.train_set = FieldpointDataset(
            self.descriptors_to_use,
            self.dataset_raw_train,
            fieldpoint_type,
            self.centers_partial_charge,
            self.centers_atom_size,
            self.centerslogP,
            radius,
        )
        # define test set
        self.test_set = FieldpointDataset(
            self.descriptors_to_use,
            self.datasetRawTest,
            fieldpoint_type,
            self.centers_partial_charge,
            self.centers_atom_size,
            self.centerslogP,
            radius,
        )

    def filter_molecule_indices(
        self, data_path, moleculeIndices, min_number_atoms, max_number_atoms
    ):
        """This function filters the moleculeIndices, to only keep those with nrAmtomsMin<#Atoms<max_number_atoms

            Args:
                data_path: path to the zarr file
                moleculeIndices: Indices of molecules to load
                min_number_atoms:  

            Return:
                moleculeDataDict: Dictionary containing the concatenated numpy arrays of all conformations
            """
        moleculeDataDict = {}
        with zarr.open(data_path, "r") as moleculeData:
            moleculeDataDict["moleculeInfo"] = moleculeData["moleculeInfo"][
                : self.number_molecules
            ]
            indConf = np.minimum(moleculeDataDict["moleculeInfo"], 1)
            beginConfInterval = np.concatenate(
                ([0], np.cumsum(moleculeDataDict["moleculeInfo"]))
            )[:-1]
            endConfInterval = indConf + beginConfInterval
            # only load conformations for certain molecules
            beginConfInterval = beginConfInterval[moleculeIndices]
            endConfInterval = endConfInterval[moleculeIndices]
            loadConfs = np.concatenate(
                [
                    np.arange(beginConfInterval[ind], endConfInterval[ind])
                    for ind in range(len(endConfInterval))
                ]
            )
            sampleIndexBegin = moleculeData["conformationInfo"][: loadConfs[-1] + 2][
                loadConfs
            ]
            sampleIndexEnd = moleculeData["conformationInfo"][: loadConfs[-1] + 2][
                loadConfs + 1
            ]
        confInfo = (sampleIndexEnd - sampleIndexBegin)[:, 1]
        restrInds = (min_number_atoms <= confInfo) & (confInfo <= max_number_atoms)
        return moleculeIndices[restrInds]

    def load_molecule_data(self, data_path, moleculeIndices):
        """
        Args:
            data_path: path to the zarr file
            moleculeIndices: Indices of molecules to load

        Return:
            moleculeDataDict: Dictionary containing the concatenated numpy arrays of all conformations
        """
        moleculeDataDict = {}
        with zarr.open(data_path, "r") as moleculeData:
            moleculeDataDict["moleculeInfo"] = moleculeData["moleculeInfo"][
                : self.number_molecules
            ]
            indConf = np.minimum(
                moleculeDataDict["moleculeInfo"], self.number_conformations
            )
            beginConfInterval = np.concatenate(
                ([0], np.cumsum(moleculeDataDict["moleculeInfo"]))
            )[:-1]
            endConfInterval = indConf + beginConfInterval
            # only load conformations for certain molecules
            beginConfInterval = beginConfInterval[moleculeIndices]
            endConfInterval = endConfInterval[moleculeIndices]
            loadConfs = np.concatenate(
                [
                    np.arange(beginConfInterval[ind], endConfInterval[ind])
                    for ind in range(len(endConfInterval))
                ]
            )
            sampleIndexBegin = moleculeData["conformationInfo"][: loadConfs[-1] + 2][
                loadConfs
            ]
            sampleIndexEnd = moleculeData["conformationInfo"][: loadConfs[-1] + 2][
                loadConfs + 1
            ]
            # sample splits for the paramsArray:
            sample_params_IndexBegin = moleculeData["paramsSplitInds"][
                : loadConfs[-1] + 2
            ][loadConfs]
            sample_params_IndexEnd = moleculeData["paramsSplitInds"][
                : loadConfs[-1] + 2
            ][loadConfs + 1]
            # atoms split points are specified in column "1"
            atomIndices = np.concatenate(
                [
                    np.arange(startInd, endInd)
                    for startInd, endInd in zip(
                        sampleIndexBegin[:, 1], sampleIndexEnd[:, 1]
                    )
                ]
            )
            moleculeDataDict["atoms"] = moleculeData["atoms"][: atomIndices[-1] + 1][
                atomIndices
            ]
            fieldpointIndices = np.concatenate(
                [
                    np.arange(startInd, endInd)
                    for startInd, endInd in zip(
                        sampleIndexBegin[:, -1], sampleIndexEnd[:, -1]
                    )
                ]
            )
            moleculeDataDict["fieldPoints"] = moleculeData["fieldPoints"][
                : fieldpointIndices[-1] + 1
            ][fieldpointIndices]
            xedIndices = np.concatenate(
                [
                    np.arange(startInd, endInd)
                    for startInd, endInd in zip(
                        sampleIndexBegin[:, 2], sampleIndexEnd[:, 2]
                    )
                ]
            )
            moleculeDataDict["xeds"] = moleculeData["xeds"][: xedIndices[-1] + 1][
                xedIndices
            ]
            edgeIndices = np.concatenate(
                [
                    np.arange(startInd, endInd)
                    for startInd, endInd in zip(
                        sampleIndexBegin[:, 0], sampleIndexEnd[:, 0]
                    )
                ]
            )
            moleculeDataDict["edges"] = moleculeData["edges"][: edgeIndices[-1] + 1][
                :, edgeIndices
            ]
            # split points for paramsArray
            paramsIndices = np.concatenate(
                [
                    np.arange(startInd, endInd)
                    for startInd, endInd in zip(
                        sample_params_IndexBegin, sample_params_IndexEnd
                    )
                ]
            )
            moleculeDataDict["paramsArray"] = moleculeData["paramsArray"][
                : paramsIndices[-1] + 1
            ][paramsIndices]
        moleculeDataDict["paramsSplitInds"] = np.concatenate(
            (
                np.array([0]),
                np.cumsum(sample_params_IndexEnd - sample_params_IndexBegin, axis=0),
            )
        )
        moleculeDataDict["conformationInfo"] = np.vstack(
            (
                np.array([0, 0, 0, 0]),
                np.cumsum(sampleIndexEnd - sampleIndexBegin, axis=0),
            )
        )
        oneHotAtomType, oneHotHybType = atomType1Hot(
            moleculeDataDict["atoms"][:, 0], moleculeDataDict["atoms"][:, 4]
        )
        moleculeDataDict["atomTypes1Hot"] = oneHotAtomType
        moleculeDataDict["atomHybTypes1Hot"] = oneHotHybType
        return moleculeDataDict


def atomType1Hot(atomTypes, atomHybTypes):
    atomTypesUnique = np.array(
        [1.0, 6.0, 7.0, 8.0, 9.0, 14.0, 15.0, 16.0, 17.0, 35.0, 53.0]
    )
    oneHotAtomType = atomTypes.reshape((-1, 1)) == atomTypesUnique
    hybTypesUnique = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
            14.0,
            15.0,
            16.0,
            17.0,
            18.0,
            19.0,
            23.0,
            24.0,
            25.0,
            26.0,
            37.0,
        ]
    )
    oneHotHybType = atomHybTypes.reshape((-1, 1)) == hybTypesUnique
    nrTypes = np.sum(oneHotAtomType, 0)
    nrHybTypes = np.sum(oneHotHybType, 0)
    print("freq of atom types", nrTypes / np.sum(nrTypes))
    nrHybTypes = np.sum(oneHotHybType, 0)
    print("freq of atom hyb types", nrHybTypes / np.sum(nrHybTypes))
    return oneHotAtomType, oneHotHybType


def num_molecules(data_path):
    """Calculate number of molecules contained in data set

    Args:
        data_path: path to data

    Return:
        union of new edges and initial edges
    """
    with zarr.open(data_path, "r") as moleculeData:
        totalNumMolecules = len(moleculeData["moleculeInfo"][:-1])
    return totalNumMolecules


def edgeConstruction(coordinates, edges, radius):
    """This function aims to construct a new graph topology
         based on distances instead of bonds.

    Args:
        coordinates: Nx3 tensor of atom positions
        edges: 2x m tensor of edge data of molecule graph
        radius: add edges between atoms that are within the ball of 'radius'
                of the atom
        keepEdges: bool value, should the bonds be kept

    Return:
        union of new edges and initial edges
    """
    coordinates = torch.tensor(coordinates)
    diffs = coordinates.reshape(
        (1, coordinates.shape[0], coordinates.shape[1])
    ) - coordinates.reshape((coordinates.shape[0], -1, coordinates.shape[1]))
    dists = torch.norm(diffs, dim=2)
    edgeMatrix = dists < radius
    edgesNew = []
    indices = torch.arange(0, len(edgeMatrix))
    for ind, row in enumerate(edgeMatrix):
        dsts = indices[ind + 1 :][row[ind + 1 :]]
        edgesNew.extend([torch.tensor([ind, dst]) for dst in dsts])
    edgesNewReshaped = torch.transpose(torch.vstack(edgesNew), 0, 1)
    return edgesNewReshaped


def collate(samples):
    """aggregate samples (merging several graphs into one)

    Args:
        samples: data samples to aggregate

    Returns:
    """
    # this function is needed to
    graphs, y, extra_data = map(list, zip(*samples))
    graphs_valid_y = []
    y_valid = []
    extra_data_valid = []
    for graph, y_it, extra_data_it in zip(graphs, y, extra_data):
        if len(y_it) > 0:
            y_valid.append(y_it)
            extra_data_valid.append(extra_data_it)
            graphs_valid_y.append(graph)
    batched_graph = dgl.batch(graphs_valid_y)
    return batched_graph, y_valid, extra_data_valid


# In original data, fieldpoints:
# blue: negative electrostatic field F- -5
# red: positive electrostatic field F+ -6
# yellow: Van der Vaals FO -7
# golden: hydrophopic FI -8
