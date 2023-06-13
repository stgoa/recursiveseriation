# encoding=utf-8

from collections import defaultdict
from functools import cache
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
        node_list: List[Qtree],
        dissimilarity: Callable,
    ):
        self.nodes = node_list  # type: List[Qtree]
        self.node_ids = [
            i for i in range(len(node_list))
        ]  # internal enumeration of the elements
        self.weights = lambda i, j: dissimilarity(
            node_list[i], node_list[j]
        )  # function
        self.components = []  # list of connected components of the graph
        self.N = len(node_list)  # number of nodes
        self.partition = None
        self.neighbourhood = defaultdict(set)  # neighbourhood mapping

        self.populate_neighbourhood_mapping()

    @cache
    def nearest_neighbours(self, node_id: int):
        """
        Compute the nearest neighbours of a node in O(N) time where N = len(weights).
        Note that this is not a symmetric relation between nodes. (i.e. if j is in the nearest neighbours of i, it does not mean that i is in the nearest neighbours of j)

        We use a cache decorator to avoid recomputing the same values.
        Parameters
        ----------
        node_id : int
            index of the node in the list of nodes
        """

        nns = None  # current arg min
        min_dist = np.inf  # current min
        for i in self.node_ids:  # nearest neight
            if i != node_id:
                dist_i = self.weights(i, node_id) # compute dissimilarity between nodes
                if (
                    dist_i < min_dist
                ):  # if dist of current is less than best known
                    min_dist = dist_i  # update current min
                    nns = {i}  # update argmin
                elif dist_i == min_dist:  # if it is as good as best know
                    nns.add(i)  # add to argmin

        return nns  # return minimal interval

    def populate_neighbourhood_mapping(self)->None:
        """
        Construct the neighbourhood function in O(N^2) time where N = len(weights).

        This is a symmetric relation between nodes.
        """

        for node1 in self.node_ids:
            for node2 in self.node_ids:
                if node1 in self.nearest_neighbours(node2) or node2 in self.nearest_neighbours(node1):
                    self.neighbourhood[node1].add(node2)
                    self.neighbourhood[node2].add(node1)

    def get_degree_one_nodes(self)->List[int]:
        """
        Compute the set of degree one nodes in O(N) time where N = len(weights)
        """
        degree_one_nodes = []
        for i in self.node_ids:
            if len(self.neighbourhood[i]) == 1:
                degree_one_nodes.append(i)
        return degree_one_nodes
    
    def depth_first_search(self, start:int, visited=None)->List[int]:
        """
        Depth first search algorithm.

        We want to keep track of the visited nodes, so that we don't visit them again. But also the order in which we visit them (since this defines an interval).

        """
        if visited is None:
            visited = []

        visited.append(start)
        for next_node in self.neighbourhood[start] - set(visited):
            visited = self.depth_first_search(next_node, visited=visited)
        return visited

    def get_DFS_order(self)->List[Qtree]:
        """
        Compute the DFS order of the graph in O(N) time where N = len(weights), starting from the degree one nodes.
        Each DFS run corresponds to a connected component of the graph (and thus a Qtree).
        This returns a list of Qtree objects in DFS order.
        """
        trees = []

        degree_one_nodes = self.get_degree_one_nodes()
        if len(degree_one_nodes) == 0: # if there are no degree one nodes, we have one large connected component
            degree_one_nodes.append(0)

        logging.debug(f"degree one nodes {degree_one_nodes}")

        all_visited = set()
        for v in degree_one_nodes:
            # run a dfs starting from each degree one node
            # each of the runs correpsonds to a connected component of the graph (and thus a Qtree)
            if v not in all_visited: 

                component_indices = self.depth_first_search(v) # get the list of nodes (index) in the component
                
                component_nodes = []
                depth = -np.inf
                for i in component_indices:
                    component_nodes.append(self.nodes[i])
                    depth = max(depth, self.nodes[i].depth)

                # generate new Q tree
                tree = Qtree(
                    children=component_nodes,
                    depth=depth + 1,
                )
                trees.append(tree)
                all_visited.update(component_indices)

        return trees
