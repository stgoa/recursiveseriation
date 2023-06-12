# encoding=utf-8
import types
from typing import List, Optional

import numpy as np

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl


Documentation pending
"""


class Qtree:
    def __init__(
        self,
        children: List,
        leave: bool = False,
        parent=None,
        depth: int = 0,
        verbose: int = 0,
    ):
        """
        Constructor of the Qtree class

        Parameters
        ----------
        children : list
            list of Qtree objects
        leave : bool, optional
            if the Qtree is a leave, by default False
        parent : Qtree, optional
            parent of the Qtree, by default None
        depth : int, optional
            depth of the Qtree, by default 0
        verbose : int, optional
            verbosity level, by default 0
        """

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

    def borders(self) -> List:
        """
        Returns the border of the Qtree

        Returns:
            List: list of elements in the border
        """
        if self.singleton:
            return self.children
        else:
            return self.left_borders() + self.right_borders()

    def left_borders(self):
        """
        Returns the left border of the Qtree
        """
        if self.singleton:
            return self.children
        return self.left_tree.borders()

    def right_borders(self):
        """
        Returns the right border of the Qtree
        """
        if self.singleton:
            return self.children
        return self.right_tree.borders()

    def frontier(self):
        """
        Returns the frontier of the Qtree
        """
        if self.singleton:
            return self.children
        else:

            frontier = []
            for child in self.children:
                frontier += child.frontier()

            return frontier

    def is_at_the_left(self, element):
        """
        Returns True if the element is at the left of the Qtree
        """
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
        """
        Inserts the Qtree in its parent Qtree (deletes a level of the Qtree)
        """
        # ojo qnodes != qtree ojo con root
        pos = self.parent.children.index(self)
        self.parent.children.pop(pos)
        aux = 0
        for child in self.children:
            child.parent = self.parent
            self.parent.children.insert(pos + aux, child)
            aux += 1

    def __repr__(self) -> str:
        """
        Returns a string representation of the Qtree
        """
        if self.singleton:
            return str(self.children[0]) + " "
        else:

            ret = "["
            for child in self.children:
                ret += child.__repr__()
            return ret + "]"

    def reverse(self):
        """
        Reverses the Qtree
        """
        self.children.reverse()
