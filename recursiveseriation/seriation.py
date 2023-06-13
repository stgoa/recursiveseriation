# encoding=utf-8
import numpy as np
import logging
from recursiveseriation.qtree import Qtree
from recursiveseriation.neighbours_graph import NearestNeighboursGraph

from typing import Callable, List, Optional, Tuple

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl


Documentation pending
"""


class RecursiveSeriation:
    """docstring for RecursiveSeriation"""

    def __init__(self, dissimilarity: Callable, n: int):
        """
        Constructor of the RecursiveSeriation class

        Parameters
        ----------
        dissimilarity : Callable
            dissimilarity function from [0, n-1] x [0, n-1] to [0, +inf]
        n : int
            number of elements
        """

        if not isinstance(dissimilarity, Callable):
            self.diss = lambda i, j: dissimilarity[i, j]
        else:
            self.diss = dissimilarity

        self.X = [
            i for i in range(n)
        ]  # list of elements to be sorted, represented by their index
        self.order = None  # seriation ordering of the elements

        self.memory_save = True

    def permute(self, array: np.ndarray, indices: np.array) -> List:
        """Compute the permutation of an array

        Args:
            array (np.ndarray): array to be permuted
            indices (np.array): permutation array

        Returns:
            np.ndarray: permuted array
        """
        if len(array) == 1:
            return array
        return [list(i) for i in np.take(array, indices, axis=0)]

    def initialize(self) -> List[Qtree]:
        """Compute the initial Q-trees, (singleton trees intially)"""
        trees = []
        for x in self.X:
            tree = Qtree(children=[x], is_singleton=True)
            trees.append(tree)
        return trees

    def ineq(
        self,
        A: List[Qtree],
        A_prime: List[Qtree],
        B: List[Qtree],
        B_prime: List[Qtree],
        z: int,
    ) -> bool:
        """Compute the max-min inequality between two sets of border (Border Candidates Orientation)"""
        return max(
            np.min([self.diss(z, a) for a in A]),
            np.min([self.diss(z, a_prime) for a_prime in A_prime]),
        ) < min(
            np.max([self.diss(z, b) for b in B]),
            np.max([self.diss(z, b_prime) for b_prime in B_prime]),
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
            A_prime (_type_): _description_
            A (_type_): _description_
            B (_type_): _description_
            B_prime (_type_): _description_

        Returns:
            _type_: _description_
        """

        for z in self.X:
            O1 = self.ineq(A, A_prime, B, B_prime, z)
            O2 = self.ineq(B, B_prime, A, A_prime, z)
            O3 = self.ineq(A, B_prime, B, A_prime, z)
            O4 = self.ineq(B, A_prime, A, B_prime, z)

            if O1 or O2:
                return "correct"
            elif O3 or O4:
                return "reverse"
        return "non-orientable"

    def internal_orientation(self, tree: Qtree) -> None:
        if len(tree.children) > 2 and tree.depth > 1:

            while not all(
                [
                    tree.children[i].is_singleton
                    for i in range(1, len(tree.children) - 1)
                ]
            ):

                for i in range(1, len(tree.children) - 1):

                    logging.debug(f"orienting the {i}-th children")

                    T_i = tree.children[i]

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

        while not all(
            [tree.children[i].is_singleton for i in range(len(tree.children))]
        ):
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

                for i in range(len(tree.children)):

                    T_i = tree.children[i]

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

    def dmin(self, tree1: Qtree, tree2: Qtree) -> Tuple[float, List]:
        argdmin = None
        current_min = np.inf
        for x in tree1.borders():
            for y in tree2.borders():
                # call the dissimilarity function between x and y
                diss_x_y = self.diss(x, y)
                if diss_x_y < current_min:
                    argdmin = [(x, y)]
                    current_min = diss_x_y
                elif diss_x_y == current_min:
                    argdmin.append((x, y))
        return current_min, argdmin

    def sort(
        self, trees: Optional[List[Qtree]] = None, iter: int = 0
    ) -> List[int]:

        if trees is None:
            trees = self.initialize()

        # we can save memory by not storing the dmins matrix (which is going to
        # be a submatrix of the dissimilarity matrix)
        dmins = []
        # this tradeoff is between memory and time

        for tree1 in trees:
            row = []
            for tree2 in trees:
                if tree2 != tree1:
                    # Compute dmin
                    dmin_val, argdmin = self.dmin(tree1, tree2)

                    logging.debug(f"argdmin {dmin_val} {argdmin}")

                    # perform an external orientation

                    for t in argdmin:
                        x, y = t
                        tree1.external_orientation(x)
                        tree2.external_orientation(y)

                    if not self.memory_save:
                        row.append(dmin_val)
                else:
                    row.append(0.0)
            if not self.memory_save:
                dmins.append(row)
        dmins = np.asarray(dmins)

        # compute the nearest neighbours graph
        if self.memory_save:
            # when we try to save memory, we dont store the dmins matrix
            # this is the case when we have a large number of elements
            # and explicitly storing the dmins matrix is not possible
            def dminw(x, y):
                val, _ = self.dmin(x, y)
                return val

            G = NearestNeighboursGraph(
                input_trees=trees,
                dissimilarity=dminw,
            )
        else:

            G = NearestNeighboursGraph(
                node_list=trees,
                dissimilarity=lambda x, y: dmins[x, y],
            )

        # obtain new trees from the set of connected componets
        new_trees = G.get_DFS_order()

        logging.info(f"iter {iter}")

        if len(new_trees) == 1:
            # perform the final internal orientation
            self.final_internal_orientation(new_trees[0])
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
