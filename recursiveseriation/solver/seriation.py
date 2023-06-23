# encoding=utf-8
from functools import cache
import numpy as np
from recursiveseriation import logger
from recursiveseriation.solver.qtree import Qtree
from recursiveseriation.solver.neighbours_graph import NearestNeighboursGraph

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

        Raises
        ------
        AssertionError
            if dissimilarity is not a binary function
            if n is not greater than 3
            if dissimilarity is not non-negative

        Attributes
        ----------
        element_dissimilarity : Callable
            dissimilarity function from [0, n-1] x [0, n-1] to [0, +inf]
        elements : list
            list of elements to be sorted, represented by their index
        tree : Qtree
            tree of all admisible seriation orderings of the elements
        memory_save : bool
            flag to save memory, by not storing the tree distance matrix (which is going to be a submatrix of the input dissimilarity matrix)

        """
        logger.warning(
            "it is up to the user to check that the dissimilarity function is symmetric and non-negative"
        )
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

        self.tree = (
            None  # tree of all admisible seriation orderings of the elements
        )
        self.memory_save = memory_save  # TODO: the cache decorator is not working with this flag
        if self.memory_save:
            logger.warning(
                f"memory_save was set {self.memory_save}, but it is not yet implemented"
            )

    def maxmin_inequality(
        self,
        A: List[Qtree],
        A_prime: List[Qtree],
        B: List[Qtree],
        B_prime: List[Qtree],
        z: int,
    ) -> bool:
        """Compute the max min inequality between two sets of border candidates (Border Candidates Orientation)

        Args:
            A (List[Qtree]): border candidates of the first set
            A_prime (List[Qtree]): border candidates of the complement of the first set
            B (List[Qtree]): border candidates of the second set
            B_prime (List[Qtree]): border candidates of the complement of the second set
            z (int): element to be compared with the border candidates

        Returns:
            bool: True if the max min inequality is satisfied, False otherwise
        """

        def dissimilarity_z(x):
            return self.element_dissimilarity(z, x)

        min_A = min(map(dissimilarity_z, A))
        min_A_prime = min(map(dissimilarity_z, A_prime))
        max_B = max(map(dissimilarity_z, B))
        max_B_prime = max(map(dissimilarity_z, B_prime))

        return max(min_A, min_A_prime) < min(max_B, max_B_prime)

    def border_candidates_orientation(
        self,
        tree: Qtree,
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
            O1 = self.maxmin_inequality(A, A_prime, B, B_prime, z)
            O2 = self.maxmin_inequality(B, B_prime, A, A_prime, z)
            O3 = self.maxmin_inequality(A, B_prime, B, A_prime, z)
            O4 = self.maxmin_inequality(B, A_prime, A, B_prime, z)

            if O1 or O2:
                tree.insert_in_parent()
                return "correct"
            elif O3 or O4:
                tree.reverse()
                tree.insert_in_parent()
                return "reverse"
        tree.insert_in_parent()
        logger.info(f"non-orientable {tree} with parent {tree.parent}")
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
                for i in range(1, len(tree.children) - 1):
                    logger.debug(f"orienting the {i}-th children of {tree}")

                    if not tree.children[i].is_singleton:
                        self.border_candidates_orientation(
                            tree=tree.children[i],
                            A_prime=(
                                tree.left_borders()
                                if i == 1
                                else tree.children[i - 1].borders()
                            ),
                            A=tree.children[i].left_borders(),
                            B=tree.children[i].right_borders(),
                            B_prime=(
                                tree.right_borders()
                                if i == len(tree.children) - 2
                                else tree.children[i + 1].borders()
                            ),
                        )
                    1 + 1

    def final_internal_orientation(self, tree: Qtree) -> None:
        """Perform the final internal orientation of a tree, by comparing the borders of the adjacent children trees

        Args:
            tree (Qtree): tree to be oriented
        """

        while not all([child.is_singleton for child in tree.children]):
            logger.debug(f"tree final {tree}")
            logger.debug(f"children {tree.children}")

            if len(tree.children) == 2:
                if not tree.children[0].is_singleton:
                    self.border_candidates_orientation(
                        tree=tree.children[0],
                        A_prime=tree.children[
                            -1
                        ].right_borders(),  # right borders of the second tree ()
                        A=tree.children[
                            0
                        ].left_borders(),  # left borders of the i-th tree
                        B=tree.children[
                            0
                        ].right_borders(),  # right borders of the i-th tree
                        B_prime=tree.children[
                            -1
                        ].left_borders(),  # left borders of the other tree
                    )

            else:
                for i in range(len(tree.children)):
                    if not tree.children[i].is_singleton:
                        self.border_candidates_orientation(
                            tree=tree.children[i],
                            A_prime=tree.children[
                                (i - 1)
                                % len(
                                    tree.children
                                )  # borders of the previous tree
                            ].borders(),
                            A=tree.children[
                                i
                            ].left_borders(),  # left borders of the i-th tree
                            B=tree.children[
                                i
                            ].right_borders(),  # right borders of the i-th tree
                            B_prime=tree.children[
                                (i + 1)
                                % len(
                                    tree.children
                                )  # borders of the next tree
                            ].borders(),
                        )

    @cache
    def tree_dissimilarity(
        self, tree1: Qtree, tree2: Qtree
    ) -> Tuple[float, List[Tuple[int, int]]]:
        """Compute the dissimilarity between two Qtrees. The dissimilarity is the minimum dissimilarity between the borders of the two trees.

        We use a cache decorator to save the dissimilarity between two trees, since it is going to be used multiple times.

        Args:
            tree1 (Qtree): first tree
            tree2 (Qtree): second tree

        Returns:
            Tuple[float, List[Tuple[int, int]]]: dissimilarity between the two trees and the list of pairs of borders that achieve the minimum dissimilarity
        """
        logger.debug(f"tree_dissimilarity {tree1}, {tree2}")
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
        logger.debug(f"argmin_element_pairs {argdmin}, min {current_min}")
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

        logger.info(f"recursion level {iter}")

        if trees is None:
            # Compute the initial Q-trees, (singleton trees intially)
            trees = [
                Qtree(children=[x], is_singleton=True) for x in self.elements
            ]

        for i, tree1 in enumerate(trees):
            # since the dissimilarity is symmetric, we only need to compute the dissimilarity between the trees that have not been compared yet
            for tree2 in trees[i + 1 :]:
                # Compute the dissimilarity between the two trees
                argmin_element_pairs = self.tree_dissimilarity(tree1, tree2)[
                    1
                ]  # type: List[Tuple[int, int]]

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
            # store the tree of all admisible orderings
            self.tree = new_trees[0]
            return self.tree.frontier()

        else:
            # perform a complete internal orientation
            for tree in new_trees:
                self.internal_orientation(tree)
            return self.sort(trees=new_trees, iter=iter + 1)


if __name__ == "__main__":
    from recursiveseriation.utils import (
        inversepermutation,
        permute,
        random_permutation,
        are_circular_orderings_same,
    )

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

    print("solved?:", are_circular_orderings_same(tau, order))
