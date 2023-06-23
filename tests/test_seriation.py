import numpy as np
from recursiveseriation.utils import (
    inversepermutation,
    permute,
    random_permutation,
    are_circular_orderings_same,
)
from recursiveseriation.solver.seriation import RecursiveSeriation


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


def test_seriation_of_distance_matrix_of_points_in_circle():
    points_in_circle = [
        [0.0, 1.0],
        [0.5, 0.8660254037844386],
        [0.8660254037844386, 0.5],
        [1.0, 0.0],
        [0.8660254037844387, -0.5],
        [0.5, -0.8660254037844384],
        [1.2246467991473532e-16, -1.0],
        [-0.5, -0.8660254037844386],
        [-0.8660254037844384, -0.5],
        [-1.0, -2.4492935982947064e-16],
        [-0.8660254037844387, 0.5],
        [-0.5, 0.8660254037844384],
    ]

    tau = random_permutation(len(points_in_circle))
    points_in_circle = permute(points_in_circle, tau)

    D = np.zeros((len(points_in_circle), len(points_in_circle)))

    for i in range(len(points_in_circle)):
        for j in range(len(points_in_circle)):
            D[i, j] = np.linalg.norm(
                np.asarray(points_in_circle[i])
                - np.asarray(points_in_circle[j])
            )

    rs = RecursiveSeriation(
        dissimilarity=lambda x, y: D[x, y],
        n=len(points_in_circle),
    )
    order = rs.sort()

    assert are_circular_orderings_same(order, inversepermutation(tau))


def test_seriation_of_large_distance_matrix_of_points_in_circle():
    # generate points in the unit circle
    points_in_circle = [
        [np.cos(2 * np.pi * x), np.sin(2 * np.pi * x)]
        for x in np.linspace(0, 1, 100)
    ]

    # randomly permute them
    pi = random_permutation(len(points_in_circle))
    points_in_circle = permute(points_in_circle, pi)

    # compute the distance matrix
    D = np.zeros((len(points_in_circle), len(points_in_circle)))

    for i in range(len(points_in_circle)):
        for j in range(len(points_in_circle)):
            D[i, j] = np.linalg.norm(
                np.asarray(points_in_circle[i])
                - np.asarray(points_in_circle[j])
            )

    # seriate the distance matrix
    rs = RecursiveSeriation(
        dissimilarity=lambda x, y: D[x, y], n=len(points_in_circle)
    )
    order = rs.sort()

    # check that the seriated order is the same as the original order
    assert are_circular_orderings_same(order, inversepermutation(pi))
