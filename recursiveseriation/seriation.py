# encoding=utf-8
from functools import cache
import numpy as np
import logging
from recursiveseriation.qtree import Qtree
from recursiveseriation.neighbours_graph import NearestNeighboursGraph

from typing import Callable, List, Optional, Tuple

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl
"""


class RecursiveSeriation:
    """Recursive seriation algorithm, based on the paper "An optimal algorithm for strict circular seriation" by Armstrong, S., Guzman, C. & Sing-Long, C. (2021)."""

    def __init__(
        self, dissimilarity: Callable, n: int, memory_save: bool = False
    ):
        """
        Constructor of the RecursiveSeriation class

        Parameters
        ----------
        dissimilarity : Callable
            dissimilarity function from [0, n-1] x [0, n-1] to [0, +inf]
        n : int
            number of elements
        memory_save : bool, optional
            flag to save memory, by not storing the tree distance matrix (which is going to be a submatrix of the input dissimilarity matrix), by default False
        """

        assert isinstance(
            dissimilarity, Callable
        ), "dissimilarity must be a binary function"  # check if dissimilarity is a function
        assert isinstance(n, int) and (
            n > 3
        ), "n must be greater than 3 for this to make sence"  # check if n is greater than 3
        assert (
            dissimilarity(n - 1, n - 1) >= 0
        ), "dissimilarity must be a binary non-negative callable"  # check if dissimilarity is non-negative

        self.element_dissimilarity = dissimilarity
        self.elements = list(
            range(n)
        )  # list of elements to be sorted, represented by their index

        self.order = None  # seriation ordering of the elements
        self.memory_save = memory_save  # TODO: the cache decorator is not working with this flag
        if self.memory_save:
            logging.warning(
                f"memory_save was set {self.memory_save}, but it is not yet implemented"
            )

    def initialize(self) -> List[Qtree]:
        """Compute the initial Q-trees, (singleton trees intially)

        Returns:
            List[Qtree]: list of singleton trees
        """
        trees = []
        for x in self.elements:
            tree = Qtree(children=[x], is_singleton=True)
            trees.append(tree)
        return trees

    def _maxmin_inequlity(
        self,
        A: List[Qtree],
        A_prime: List[Qtree],
        B: List[Qtree],
        B_prime: List[Qtree],
        z: int,
    ) -> bool:
        """Compute the max min inequlity between two sets of border candidates (Border Candidates Orientation)

        Args:
            A (List[Qtree]): border candidates of the first set
            A_prime (List[Qtree]): border candidates of the complement of the first set
            B (List[Qtree]): border candidates of the second set
            B_prime (List[Qtree]): border candidates of the complement of the second set
            z (int): element to be compared with the border candidates

        Returns:
            bool: True if the max min inequlity is satisfied, False otherwise
        """
        return max(
            np.min([self.element_dissimilarity(z, a) for a in A]),
            np.min(
                [self.element_dissimilarity(z, a_prime) for a_prime in A_prime]
            ),
        ) < min(
            np.max([self.element_dissimilarity(z, b) for b in B]),
            np.max(
                [self.element_dissimilarity(z, b_prime) for b_prime in B_prime]
            ),
        )

    def border_candidates_orientation(
        self,
        A_prime: List[Qtree],
        A: List[Qtree],
        B: List[Qtree],
        B_prime: List[Qtree],
    ) -> str:
        """Given border candidates A and B, of an interval I, and border candidates A_prime, B_prime of the complement I^c, this procedure determines if
        I must be fixed, reversed or if it is not orientable.

        Args:
            A_prime (List[Qtree]): border candidates of the complement of I
            A (List[Qtree]): border candidates of I
            B (List[Qtree]): border candidates of I
            B_prime (List[Qtree]): border candidates of the complement of I

        Returns:
            str: "correct" if I is correct, "reverse" if I must be reversed, "non-orientable" if I is not orientable
        """

        for z in self.elements:
            O1 = self._maxmin_inequlity(A, A_prime, B, B_prime, z)
            O2 = self._maxmin_inequlity(B, B_prime, A, A_prime, z)
            O3 = self._maxmin_inequlity(A, B_prime, B, A_prime, z)
            O4 = self._maxmin_inequlity(B, A_prime, A, B_prime, z)

            if O1 or O2:
                return "correct"
            elif O3 or O4:
                return "reverse"
        return "non-orientable"

    def internal_orientation(self, tree: Qtree) -> None:
        """Perform an internal orientation of a tree, by comparing the borders of the adjacent children trees

        Args:
            tree (Qtree): tree to be oriented
        """
        if len(tree.children) > 2 and tree.depth > 1:

            while not all(
                [child.is_singleton for child in tree.children[1:-1]]
            ):

                for i, T_i in enumerate(tree.children[1:-1]):

                    logging.debug(f"orienting the {i}-th children")

                    if not T_i.is_singleton:

                        A = T_i.left_borders()
                        B = T_i.right_borders()

                        if i == 1:
                            A_prime = tree.left_borders()
                        else:
                            A_prime = tree.children[i - 1].borders()

                        if i == len(tree.children) - 2:
                            B_prime = tree.right_borders()
                        else:
                            B_prime = tree.children[i + 1].borders()

                        orientation = self.border_candidates_orientation(
                            A_prime, A, B, B_prime
                        )

                        if orientation in ["non-orientable", "correct"]:
                            T_i.insert_in_parent()
                        else:
                            T_i.reverse()
                            T_i.insert_in_parent()

    def final_internal_orientation(self, tree: Qtree) -> None:
        """Perform the final internal orientation of a tree, by comparing the borders of the adjacent children trees

        Args:
            tree (Qtree): tree to be oriented
        """

        while not all([child.is_singleton for child in tree.children]):
            logging.debug(f"tree final {tree}")
            logging.debug(f"children {tree.children}")

            if len(tree.children) == 2:

                T_i = tree.children[0]

                if not T_i.is_singleton:

                    A_prime = tree.children[-1].right_borders()
                    A = T_i.left_borders()
                    B = T_i.right_borders()
                    B_prime = tree.children[-1].left_borders()

                    orientation = self.border_candidates_orientation(
                        A_prime, A, B, B_prime
                    )

                    if orientation in ["non-orientable", "correct"]:
                        T_i.insert_in_parent()
                    else:
                        T_i.reverse()
                        T_i.insert_in_parent()

                    tree.children[-1].insert_in_parent()

            else:

                for i, T_i in enumerate(tree.children):

                    if not T_i.is_singleton:

                        A_prime = tree.children[
                            (i - 1) % len(tree.children)
                        ].borders()
                        A = T_i.left_borders()
                        B = T_i.right_borders()
                        B_prime = tree.children[
                            (i + 1) % len(tree.children)
                        ].borders()

                        orientation = self.border_candidates_orientation(
                            A_prime, A, B, B_prime
                        )

                        if orientation in ["non-orientable", "correct"]:
                            T_i.insert_in_parent()
                        else:
                            T_i.reverse()
                            T_i.insert_in_parent()

    @cache
    def tree_dissimilarity(
        self, tree1: Qtree, tree2: Qtree
    ) -> Tuple[float, List]:
        """Compute the dissimilarity between two Qtrees. The dissimilarity is the minimum dissimilarity between the borders of the two trees.

        We use a cache decorator to save the dissimilarity between two trees, since it is going to be used multiple times.

        Args:
            tree1 (Qtree): first tree
            tree2 (Qtree): second tree

        Returns:
            Tuple[float, List]: dissimilarity between the two trees and the list of pairs of borders that achieve the minimum dissimilarity
        """
        argdmin = None
        current_min = np.inf
        for x in tree1.borders():
            for y in tree2.borders():
                # call the dissimilarity function between x and y
                diss_x_y = self.element_dissimilarity(x, y)
                if diss_x_y < current_min:
                    argdmin = [(x, y)]
                    current_min = diss_x_y
                elif diss_x_y == current_min:
                    argdmin.append((x, y))
        logging.debug(f"argmin_element_pairs {argdmin}")
        return current_min, argdmin

    def sort(
        self, trees: Optional[List[Qtree]] = None, iter: int = 0
    ) -> List[int]:
        """Sort the elements using the recursive seriation algorithm

        Args:
            trees (Optional[List[Qtree]], optional): list of trees to be sorted. If None, the trees are initialized. Defaults to None.
            iter (int, optional): iteration number (recursion depth). Defaults to 0. (only for logging purposes)

        Returns:
            List[int]: seriation ordering of the elements
        """

        logging.info(f"iter {iter}")

        # initialize the trees
        if trees is None:
            trees = self.initialize()

        for tree1 in trees:
            for tree2 in trees:
                if tree2 != tree1:
                    # Compute the dissimilarity between the two trees
                    argmin_element_pairs = self.tree_dissimilarity(
                        tree1, tree2
                    )[1]

                    # perform an external orientation
                    for x, y in argmin_element_pairs:
                        tree1.external_orientation(x)
                        tree2.external_orientation(y)

        # compute the nearest neighbours graph
        G = NearestNeighboursGraph(
            input_trees=trees,
            dissimilarity=lambda x, y: self.tree_dissimilarity(x, y)[0],
        )

        # obtain new trees from the set of connected componets
        new_trees = G.get_Qtrees_from_components()

        if len(new_trees) == 1:
            # perform the final internal orientation
            self.final_internal_orientation(new_trees[0])
            # store the seriation ordering
            self.order = new_trees[0].frontier()
            return self.order

        else:
            # perform a complete internal orientation
            for tree in new_trees:
                self.internal_orientation(tree)
            return self.sort(trees=new_trees, iter=iter + 1)


if __name__ == "__main__":
    from utils import inversepermutation, permute, random_permutation

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
    ]

    np.set_printoptions(precision=2)

    D = np.asarray(R)
    n = len(D)

    # np.random.seed(124)

    pi = random_permutation(len(D))  # Permutation
    D = permute(D, pi)

    rs = RecursiveSeriation(
        dissimilarity=lambda x, y: D[x, y],
        n=n,
    )
    order = rs.sort()

    tau = inversepermutation(pi)

    print(tau, order)
