# encoding=utf-8
import numpy as np

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl
"""


def inversepermutation(arr: np.array) -> np.array:
    """Computes the inverse permutation of a given permutation array

    Args:
        arr (np.array): permutation array

    Returns:
        np.array: inverse permutation array
    """
    N = len(arr)
    arr2 = [0 for _ in range(N)]
    for i in range(0, N):
        arr2[arr[i]] = i
    return arr2


def inpermute(array: np.ndarray, indices: np.array) -> np.ndarray:
    """Permutes the rows and columns of a matrix according to the inverse of a given permutation array

    Args:
        array (np.ndarray): matrix to be permuted
        indices (np.array): permutation array

    Returns:
        np.ndarray: permuted matrix
    """
    indices = inversepermutation(indices)
    return permute(array, indices)


def permute(array: np.ndarray, indices: np.array) -> np.ndarray:
    """Permutes the rows and columns of a matrix according to a given permutation array

    Args:
        array (np.ndarray): matrix to be permuted
        indices (np.array): permutation array

    Returns:
        np.ndarray: permuted matrix
    """
    array = np.asarray(array)[indices]
    if len(array.shape) == 2 and array.shape[0] == array.shape[1]:
        for idx in range(array.shape[0]):
            array[idx] = array[idx][indices]
    return array


def random_permutation(N: int) -> np.array:
    """Computes a random permutation array of size N

    Args:
        N (int): size of the permutation array

    Returns:
        np.array: random permutation array
    """
    pi = np.arange(0, N)
    np.random.shuffle(pi)
    return pi


def are_circular_orderings_same(list1: list, list2: list) -> bool:
    """
    Determines if two lists are the same up to circular permutations or reversals.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        bool: True if the lists are the same up to circular permutations or reversals, False otherwise.
    """
    if len(list1) != len(list2):
        return False

    circular_permutation = list1 + list1
    circular_permutation_str = " ".join(map(str, circular_permutation))
    list2_str = " ".join(map(str, list2))

    if list2_str in circular_permutation_str:
        return True

    reversed_circular_permutation = list1[::-1] + list1[::-1]
    reversed_circular_permutation_str = " ".join(
        map(str, reversed_circular_permutation)
    )

    if list2_str in reversed_circular_permutation_str:
        return True

    return False
