import numpy as np
from recursiveseriation.utils import (
    inversepermutation,
    permute,
    random_permutation,
    are_circular_orderings_same,
)
from recursiveseriation.seriation import RecursiveSeriation


def test_seriation():
    # Test case 1: Permute a strict circular Robinson matrix with a valid
    # permutation array for rows and columns
    R = [
        [0, 1, 3, 5, 6, 7, 7, 6, 5, 4, 3],
        [1, 0, 2, 4, 5, 6, 7, 7, 6, 5, 4],
        [3, 2, 0, 1, 4, 5, 6, 7, 7, 6, 5],
        [5, 4, 1, 0, 1, 4, 5, 6, 7, 7, 6],
        [6, 5, 4, 1, 0, 1, 4, 5, 6, 7, 7],
        [7, 6, 5, 4, 1, 0, 3, 4, 5, 6, 7],
        [7, 7, 6, 5, 4, 3, 0, 1, 4, 5, 6],
        [6, 7, 7, 6, 5, 4, 1, 0, 2, 4, 5],
        [5, 6, 7, 7, 6, 5, 4, 2, 0, 1, 4],
        [4, 5, 6, 7, 7, 6, 5, 4, 1, 0, 1],
        [3, 4, 5, 6, 7, 7, 6, 5, 4, 1, 0],
    ]  # Some strict circular Robinson matrix

    np.set_printoptions(precision=2)

    D = np.asarray(R)
    n = len(D)

    pi = random_permutation(len(D))  # Permutation
    D = permute(D, pi)

    rs = RecursiveSeriation(
        dissimilarity=lambda x, y: D[x, y],
        n=n,
    )
    order = rs.sort()

    tau = inversepermutation(pi)

    assert are_circular_orderings_same(tau, order)
