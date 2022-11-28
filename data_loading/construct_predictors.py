"""This module contains functions for constructing descriptors"""
############################################
import numpy as np


def atom_size_from_num(atom_nums):
    """Provide atom size for given atomic number

        Args:
            atom_nums: atomic number

        Return: 
            atom_sizes[atom_nums]: atom sizes corresponding to atomic numbers
                
    """
    atom_sizes = np.array(
        [
            0.0,
            1.2,
            1.4,
            2.2,
            1.9,
            1.8,
            1.7,
            1.6,
            1.55,
            1.5,
            1.54,
            2.4,
            2.2,
            2.1,
            2.1,
            1.95,
            1.8,
            1.8,
            1.88,
            2.8,
            2.4,
            2.3,
            2.15,
            2.05,
            2.05,
            2.05,
            2.05,
            2.0,
            2.0,
            2.0,
            2.1,
            2.1,
            2.1,
            2.05,
            1.9,
            1.9,
            2.02,
            2.9,
            2.55,
            2.4,
            2.3,
            2.15,
            2.1,
            2.05,
            2.05,
            2.0,
            2.05,
            2.1,
            2.2,
            2.2,
            2.25,
            2.2,
            2.1,
            2.1,
            2.16,
            3.0,
            2.7,
            2.5,
            2.48,
            2.47,
            2.45,
            2.43,
            2.42,
            2.4,
            2.38,
            2.37,
            2.35,
            2.33,
            2.32,
            2.3,
            2.28,
            2.27,
            2.25,
            2.2,
            2.1,
            2.05,
            2.0,
            2.0,
            2.05,
            2.1,
            2.05,
            2.2,
            2.3,
            2.3,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.4,
            2.0,
            2.3,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ]
    )
    return atom_sizes[atom_nums]


def rbf_expand(vector, centers):
    """Produce a radial basis function expansion

        Args:
            vector: the vector of scalars to expand
            centers: the support points to use in the expansion 

        Return:
                vecExpanded: The rbf expanded matrix (dimension: len(vector) x #centers)
    """
    sigma = np.abs(centers[1] - centers[0]) ** 0.5
    vecExpanded = np.exp(
        -(sigma ** (-2)) * (vector.reshape((-1, 1)) - centers.reshape((1, -1))) ** 2
    )
    return vecExpanded


if __name__ == "__main__":
    atom_nums = [1, 2, 4]
    atom_size_from_num(atom_nums)
