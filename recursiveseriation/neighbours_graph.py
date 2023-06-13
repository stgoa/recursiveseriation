# encoding=utf-8

from typing import Callable, List
import logging
import numpy as np

from recursiveseriation.qtree import Qtree

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl


Documentation pending
"""


class NNGraph:
    def __init__(
        self,
        node_list: List,
        dissimilarity: Callable,
    ):
        self.nodes = node_list  # list of nodes
        self.node_ids = [
            i for i in range(len(node_list))
        ]  # internal enumeration of the elements
        self.weights = lambda i, j: dissimilarity(
            node_list[i], node_list[j]
        )  # function
        self.components = []  # list of connected components of the graph
        self.N = len(node_list)  # number of nodes
        self.borders = []
        self.partition = None
        self.neighbourhood = {
            i: set() for i in self.node_ids
        }  # neighbourhood function

        self.get_neighbours()

    def nearest_neighbours(self, node_id: int):
        """
        Compute the nearest neighbours of a node in O(N) time where N = len(weights)

        Parameters
        ----------
        node_id : int
            index of the node in the list of nodes
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
        Construct the neighbourhood function in O(N^2) time where N = len(weights)
        """

        dir_edges = [self.nearest_neighbours(node) for node in self.node_ids]

        for n1 in self.node_ids:
            for n2 in self.node_ids:
                if n1 in dir_edges[n2] or n2 in dir_edges[n1]:
                    self.neighbourhood[n1].add(n2)
                    self.neighbourhood[n2].add(n1)

    def get_degree_one_nodes(self):
        """
        Compute the set of degree one nodes in O(N) time where N = len(weights)
        """

        for i in self.node_ids:
            if len(self.neighbourhood[i]) == 1:
                self.borders.append(i)

    def depth_first_search(self, start, visited=[]):
        """
        Depth first search algorithm
        """
        visited.append(start)
        for next_node in self.neighbourhood[start] - set(visited):
            visited = self.depth_first_search(next_node, visited=visited)
            if len(self.neighbourhood[start] - set(visited)) == 0:
                break
        return visited

    def get_DFS_order(self):
        """
        Compute the DFS order of the graph in O(N) time where N = len(weights)
        """
        trees = []

        self.get_degree_one_nodes()
        if len(self.borders) == 0:
            self.borders.append(0)

        logging.debug(f"degree one nodes {self.borders}")

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
                )

                trees.append(tree)

                visited += component

        return trees
