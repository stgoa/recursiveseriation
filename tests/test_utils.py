import numpy as np

from recursiveseriation.utils import (
    are_circular_orderings_same,
    inpermute,
    permute,
    random_permutation,
)


def test_are_circular_orderings_same():
    # Same list, not reversed or circularly permuted
    list1 = [1, 2, 3, 4]
    list2 = [1, 2, 3, 4]
    assert are_circular_orderings_same(list1, list2)

    # Same list, circularly permuted
    list1 = [1, 2, 3, 4]
    list2 = [4, 1, 2, 3]
    assert are_circular_orderings_same(list1, list2)

    # Same list, reversed
    list1 = [1, 2, 3, 4]
    list2 = [4, 3, 2, 1]
    assert are_circular_orderings_same(list1, list2)

    # Different lists
    list1 = [1, 2, 3, 4]
    list2 = [5, 6, 7, 8]
    assert are_circular_orderings_same(list1, list2) == False

    # Lists with different lengths
    list1 = [1, 2, 3, 4]
    list2 = [1, 2, 3]
    assert are_circular_orderings_same(list1, list2) == False

    # Same list, circularly permuted and reversed
    list1 = [1, 2, 3, 4]
    list2 = [3, 4, 1, 2]
    assert are_circular_orderings_same(list1, list2)

    # Same list, reversed multiple times
    list1 = [1, 2, 3, 4]
    list2 = [2, 1, 4, 3]
    assert are_circular_orderings_same(list1, list2)

    # Same list, circularly permuted multiple times
    list1 = [1, 2, 3, 4]
    list2 = [4, 3, 2, 1]
    assert are_circular_orderings_same(list1, list2)


def test_permute():
    # Test case 1: Permute a 3x3 matrix with a valid permutation array for
    # rows and columns
    array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    indices = np.array([2, 0, 1])
    expected_result = np.array([[9, 7, 8], [3, 1, 2], [6, 4, 5]])
    assert np.array_equal(permute(array, indices), expected_result)

    # Test case 2: Permute a 4x4 matrix with a valid permutation array for
    # rows and columns
    array = np.array(
        [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]
    )
    indices = np.array([3, 1, 0, 2])
    expected_result = np.array(
        [[16, 14, 13, 15], [8, 6, 5, 7], [4, 2, 1, 3], [12, 10, 9, 11]]
    )
    assert np.array_equal(permute(array, indices), expected_result)

    # Test case 3: Permute a 2x2 matrix with identity permutation array
    array = np.array([[1, 2], [3, 4]])
    indices = np.array([0, 1])
    expected_result = np.array(
        [[1, 2], [3, 4]]
    )  # The matrix remains unchanged
    assert np.array_equal(permute(array, indices), expected_result)


def test_inpermute():
    # this simulaneously tests permute, inpermute and random_permutation
    permutation = random_permutation(10)
    array = np.random.rand(10, 10)
    # Test case 1: Inverse permutation, the inverse of a permutation is the
    # permutation that undoes the permutation
    assert np.array_equal(
        inpermute(permute(array, permutation), permutation), array
    )
