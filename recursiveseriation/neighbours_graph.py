# encoding=utf-8
import numpy as np
import types
from recursiveseriation.qtree import Qtree

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl


Documentation pending
"""


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
