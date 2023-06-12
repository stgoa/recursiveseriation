# encoding=utf-8
import numpy as np
import types

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl


Documentation pending
"""


def inversepermutation(arr):
    N = len(arr)
    arr2 = [0 for i in range(N)]
    for i in range(0, N):
        arr2[arr[i]] = i
    return arr2


def inpermute(array, indices):
    indices = inversepermutation(indices)
    return permute(array, indices)


def permute(array, indices):
    array = np.asarray(array)[indices]
    if len(array.shape) == 2:
        for idx in range(array.shape[0]):
            array[idx] = array[idx][indices]
    return array


def random_permutation(N):
    pi = np.arange(0, N)
    np.random.shuffle(pi)
    return pi


class RecursiveSeriation:
    """docstring for RecursiveSeriation"""

    def __init__(self, dissimilarity, n, verbose=0):
        """
        dissimilarity : binary function"""

        if not isinstance(dissimilarity, types.FunctionType):
            self.diss = lambda i, j: dissimilarity[i, j]
        else:
            self.diss = dissimilarity

        self.X = [i for i in range(n)]

        self.verbose = verbose
        self.order = None

        self.memory_save = True

    def permute(self, array, indices):
        if len(array) == 1:
            return array
        return [list(i) for i in np.take(array, indices, axis=0)]

    def initial(self):
        trees = []
        for x in self.X:
            tree = Qtree(children=[x], leave=True)
            tree.singleton = True
            trees.append(tree)
        return trees

    def ineq(self, A, A_prime, B, B_prime, z):
        return max(
            np.min([self.diss(z, a) for a in A]),
            np.min([self.diss(z, a_prime) for a_prime in A_prime]),
        ) < min(
            np.max([self.diss(z, b) for b in B]),
            np.max([self.diss(z, b_prime) for b_prime in B_prime]),
        )

    def border_candidates_orientation(self, A_prime, A, B, B_prime):

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

    def internal_orientation(self, tree):
        if len(tree.children) > 2 and tree.depth > 1:

            while not all(
                [
                    tree.children[i].singleton
                    for i in range(1, len(tree.children) - 1)
                ]
            ):

                for i in range(1, len(tree.children) - 1):

                    if self.verbose > 0:
                        print("orienting the ", i, "-th children")

                    T_i = tree.children[i]

                    if not T_i.singleton:

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

    def final_internal_orientation(self, tree):

        while not all(
            [tree.children[i].singleton for i in range(len(tree.children))]
        ):
            if self.verbose > 3:
                print("tree final", tree)
                print("children", tree.children)

            if len(tree.children) == 2:
                # print("FINAL ORIENTATION")

                T_i = tree.children[0]

                if not T_i.singleton:

                    A_prime = tree.children[-1].right_borders()
                    A = T_i.left_borders()
                    B = T_i.right_borders()
                    B_prime = tree.children[-1].left_borders()

                    orientation = self.border_candidates_orientation(
                        A_prime, A, B, B_prime
                    )

                    if self.verbose > 0:
                        print("OAE", tree)
                        print("OAE first", tree.children[0])
                        print("OAE second", tree.children[-1])
                        print("orientation at end", orientation)

                    if orientation in ["non-orientable", "correct"]:
                        T_i.insert_in_parent()
                    else:
                        T_i.reverse()
                        T_i.insert_in_parent()

                    tree.children[-1].insert_in_parent()
                    if self.verbose > 0:
                        print("post", tree)

                    if self.verbose > 3:

                        print(
                            "break condition",
                            [
                                (tree.children[i], tree.children[i].singleton)
                                for i in range(len(tree.children))
                            ],
                        )

            else:
                if self.verbose > 3:
                    print("NOT FINAL")
                for i in range(len(tree.children)):
                    if self.verbose > 1:
                        print("orienting the ", i, "-th children")

                    T_i = tree.children[i]

                    if not T_i.singleton:

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

    def dmin(self, tree1, tree2):
        argdmin = None
        current_min = np.inf
        for x in tree1.borders():
            for y in tree2.borders():
                if self.diss(x, y) < current_min:
                    argdmin = [(x, y)]
                    current_min = self.diss(x, y)
                elif self.diss(x, y) == current_min:
                    argdmin.append((x, y))
        return current_min, argdmin

    def sort(self, trees=None, iter=0):

        if trees is None:
            trees = self.initial()

        dmins = []

        for tree1 in trees:
            row = []
            for tree2 in trees:
                if tree2 != tree1:
                    # Compute dmin
                    dmin_val, argdmin = self.dmin(tree1, tree2)

                    if self.verbose > 0:

                        print("argdmin", dmin_val, argdmin)

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

        if self.verbose > 0:
            print(dmins)

        # compute the nearest neighbours graph

        if self.memory_save:

            def dminw(x, y):
                val, _ = self.dmin(x, y)
                return val

            G = NNGraph(
                node_list=trees, dissimilarity=dminw, verbose=self.verbose
            )
        else:

            G = NNGraph(
                node_list=trees,
                dissimilarity=lambda x, y: dmins[x, y],
                verbose=self.verbose,
            )

        # obtain new trees from the set of connected componets
        new_trees = G.get_DFS_order()

        print("iter", iter)

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


class Qtree:
    def __init__(
        self, children=[], leave=False, parent=None, depth=0, verbose=0
    ):

        self.singleton = leave
        self.left_tree = children[0]
        self.right_tree = children[-1]
        self.depth = depth

        self.verbose = verbose

        # atributes of the root
        self.children = children
        self.non_orientable = False
        self.oriented = False
        # self.leave = leave
        self.parent = parent

        if not leave:
            for child in children:
                child.parent = self

    def borders(self):
        if self.singleton:
            return self.children
        else:
            return self.left_borders() + self.right_borders()

    def left_borders(self):
        if self.singleton:
            return self.children
        return self.left_tree.borders()

    def right_borders(self):
        if self.singleton:
            return self.children
        return self.right_tree.borders()

    def frontier(self):
        if self.singleton:
            return self.children
        else:

            frontier = []
            for child in self.children:
                frontier += child.frontier()

            return frontier

    def is_at_the_left(self, element):
        return element in self.left_tree.frontier()

    def external_orientation(self, element):

        if not self.singleton:

            if self.verbose > 0:
                print("external orientation of", self, "at element", element)

            if self.is_at_the_left(element):

                current = self.left_tree

                if self.verbose > 1:
                    print("current", current)
                    print("current.parent", current.parent)
                    print("current.parent.children", current.parent.children)

                while not current.singleton and not current.oriented:
                    current.oriented = True

                    if current.is_at_the_left(element):
                        current.insert_in_parent()
                    else:
                        current.reverse()
                        current.insert_in_parent()

                    current = current.left_tree
            else:

                current = self.right_tree

                while not current.singleton and not current.oriented:
                    current.oriented = True

                    if current.is_at_the_left(element):
                        current.reverse()
                        current.insert_in_parent()
                    else:
                        current.insert_in_parent()

                    current = current.right_tree

    def insert_in_parent(self):
        # ojo qnodes != qtree ojo con root
        pos = self.parent.children.index(self)
        self.parent.children.pop(pos)
        aux = 0
        for child in self.children:
            child.parent = self.parent
            self.parent.children.insert(pos + aux, child)
            aux += 1

    def __repr__(self):
        if self.singleton:
            return str(self.children[0]) + " "
        else:

            ret = "["
            for child in self.children:
                ret += child.__repr__()
            return ret + "]"

    def reverse(self):
        self.children.reverse()


class NNGraph:
    def __init__(self, node_list, dissimilarity, verbose=0):
        self.nodes = node_list
        self.node_ids = [
            i for i in range(len(node_list))
        ]  # internal enumeration of the elements
        self.weights = lambda i, j: dissimilarity(
            node_list[i], node_list[j]
        )  # function
        self.components = []
        self.N = len(node_list)
        self.borders = []
        self.partition = None
        self.neighbourhood = {i: set() for i in self.node_ids}
        self.verbose = verbose

        self.get_neighbours()

    def nearest_neighbours(self, node_id):
        """
        Complexity = O(N)
        where N = len(weights)
        """

        nns = []  # current arg min
        min_dist = self.weights(
            node_id, (node_id + 1) % self.N
        )  # nearest neighbour

        for i in self.node_ids:  # nearest neight
            if i != node_id:
                dist_i = self.weights(i, node_id)
                if (
                    dist_i < min_dist
                ):  # if dist of current is less than best known
                    min_dist = dist_i  # update current min
                    nns = [i]  # update argmin
                elif dist_i == min_dist:  # if it is as good as best know
                    nns.append(i)  # add to argmin

        return nns  # return minimal interval

    def get_neighbours(self):

        """
        Construct the neighbourhood function

        Complexity = O(N^2)
        where N = len(weights)
        """

        dir_edges = [self.nearest_neighbours(node) for node in self.node_ids]

        for n1 in self.node_ids:
            for n2 in self.node_ids:
                if n1 in dir_edges[n2] or n2 in dir_edges[n1]:
                    self.neighbourhood[n1].add(n2)
                    self.neighbourhood[n2].add(n1)

    def get_degree_one_nodes(self):
        """
        Compute the set of degree one nodes
        Complexity: O(N)"""

        for i in self.node_ids:
            if len(self.neighbourhood[i]) == 1:
                self.borders.append(i)

    def depth_first_search(self, start, visited=[]):
        visited.append(start)
        for next_node in self.neighbourhood[start] - set(visited):
            visited = self.depth_first_search(next_node, visited=visited)
            if len(self.neighbourhood[start] - set(visited)) == 0:
                break
        return visited

    def get_DFS_order(self):
        """Complexity: O(N)"""
        trees = []

        self.get_degree_one_nodes()
        if len(self.borders) == 0:
            self.borders.append(0)

        if self.verbose > 0:

            print("degree one nodes", self.borders)

        visited = []

        for v in self.borders:

            if v not in visited:

                # print("starting at border {}".format(v))

                component = self.depth_first_search(v, visited=[])

                # generate new Q tree
                pre_depth = np.max([self.nodes[i].depth for i in component])
                tree = Qtree(
                    children=[self.nodes[i] for i in component],
                    depth=pre_depth + 1,
                    verbose=self.verbose,
                )

                trees.append(tree)

                visited += component

                if self.verbose > 2:

                    print("new component added: ", component)
                    print("borders")
                    print(
                        tree.left_borders(),
                        tree.right_borders(),
                        tree.borders(),
                    )
                    print("left tree", tree.left_tree)
                    print("left tree's parent", tree.left_tree.parent)
                    print("frontier", tree.frontier())
                    print("representation", tree)

        return trees


if __name__ == "__main__":

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

    verbose = 0

    rs = RecursiveSeriation(
        dissimilarity=lambda x, y: D[x, y], n=n, verbose=verbose
    )
    order = rs.sort()

    tau = inversepermutation(pi)

    print(tau, order)
