# encoding=utf-8
from typing import List
from recursiveseriation import logger

"""
Author: Santiago Armstrong
email: sarmstrong@uc.cl
"""


class Qtree:
    def __init__(
        self,
        children: List,
        is_singleton: bool = False,
        parent=None,
        depth: int = 0,
    ):
        """
        Constructor of the Qtree class

        Parameters
        ----------
        children : list
            list of Qtree objects
        is_singleton : bool, optional
            if the Qtree is a singleton (tree leaf), by default False
        parent : Qtree, optional
            parent of the Qtree, by default None
        depth : int, optional
            depth of the Qtree, by default 0

        Attributes
        ----------
        is_singleton : bool
            if the Qtree is a singleton (tree leaf)
        left_tree : Qtree
            left subtree
        right_tree : Qtree
            right subtree
        depth : int
            depth of the Qtree (root is 0)
        children : list
            list of children Qtree objects
        oriented : bool
            True if the Qtree is oriented, False otherwise
        parent : Qtree
            parent of the Qtree
        """

        self.is_singleton = is_singleton
        self.left_tree = children[0]  # left subtree
        self.right_tree = children[-1]  # right subtree
        self.depth = depth  # depth of the Qtree (root is 0)

        # atributes of the root
        self.children = children
        # self.non_orientable = False
        self.oriented = False
        self.parent = parent

        if not is_singleton:
            for child in children:
                child.parent = self

    def borders(self) -> List:
        """
        Returns the border of the Qtree

        Returns:
            List: list of elements in the border
        """
        if self.is_singleton:
            return self.children
        return self.left_borders() + self.right_borders()

    def left_borders(self):
        """
        Returns the left border of the Qtree
        """
        if self.is_singleton:
            return self.children
        return self.left_tree.borders()

    def right_borders(self):
        """
        Returns the right border of the Qtree
        """
        if self.is_singleton:
            return self.children
        return self.right_tree.borders()

    def frontier(self) -> List[int]:
        """
        Returns the frontier of the Qtree in any admisible order
        """
        if self.is_singleton:
            return self.children
        frontier = []
        for child in self.children:
            frontier += child.frontier()
        return frontier

    def is_at_the_left(self, element: int) -> bool:
        """
        Returns True if the element is in the left subtree of the Qtree
        """
        return element in self.left_tree.frontier()

    def external_orientation(self, element: int):
        """
        Computes the external orientation of the Qtree, using the element as reference

        Parameters
        ----------
        element : int
            element known to be at a border of the interval represented by the tree, used to compute the orientation of the tree
        """

        if not self.is_singleton:
            logger.debug(
                f"external orientation of {self} at element {element}"
            )

            current, need_to_go_left = (
                (self.left_tree, True)
                if self.is_at_the_left(element)
                else (self.right_tree, False)
            )

            while not current.is_singleton and not current.oriented:
                current.oriented = True

                is_at_the_left = current.is_at_the_left(element)

                if (not is_at_the_left and need_to_go_left) or (
                    is_at_the_left and not need_to_go_left
                ):
                    current.reverse()

                current.insert_in_parent()
                current = (
                    current.left_tree
                    if need_to_go_left
                    else current.right_tree
                )

    def insert_in_parent(self):
        """
        Inserts the Qtree in its parent Qtree (deletes a level of the Qtree)

        TODO: this can be improved with a linked list, instead of a list
        the methods index and insert are O(n) in a regular Python list
        """
        pos = self.parent.children.index(
            self
        )  # position of the Qtree in the parent
        self.parent.children.pop(pos)  # delete the Qtree from the parent
        for i, child in enumerate(self.children):
            child.parent = self.parent  # change the parent of the children
            self.parent.children.insert(
                pos + i, child
            )  # insert the children in the parent preserving the order

    def __repr__(self) -> str:
        """
        Returns a string representation of the Qtree
        """
        if self.is_singleton:
            return str(self.children[0]) + " "
        ret = "["
        for child in self.children:
            ret += child.__repr__()
        return ret + "]"

    def reverse(self):
        """
        Reverses the Qtree
        """
        self.children.reverse()
        # change left and right
        aux = self.left_tree
        self.left_tree = self.right_tree
        self.right_tree = aux
