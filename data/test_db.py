from typing import Tuple

from rich.tree import Tree
from rich import print as rprint

from enum import Enum

from collections import deque

import os


class DNodeTypes(Enum):
    ROOT = "root"
    FUNCTION = "f"
    INPUT_SET = "is"
    OUTPUT_SET = "os"
    ANALYZE_SET = "as"
    PLOT = "p"


class DNode:

    def __init__(self, name: str, node_type: DNodeTypes = DNodeTypes.ROOT, *args, **kwargs) -> None:
        self.__name = name
        self.__type: DNodeTypes = node_type
        self.__build_params: dict = kwargs.get('build_params', dict())
        self.__path = kwargs.get('path', f"./{name}")

        self.children: list = []

    def add(self, child) -> None:
        child.path = f"{self.path}/{child.name}"
        self.children.append(child)

    @property
    def name(self) -> str:
        return self.__name

    @property
    def path(self) -> str:
        return self.__path

    @path.setter
    def path(self, value: str) -> None:
        self.__path = value if value != "" else self.__path

    def __str__(self):
        return f"\nType: {self.__type.name}\n Build: {self.__build_params}\n Path: {self.__path}\n"


class DataCenter:

    def __init__(self, *args, **kwargs) -> None:
        """
        Facade for managing test data for PCI
        """

        super().__init__(*args, **kwargs)

        # root path
        self.__path = kwargs.get('path', "./")
        # create NTree from directories
        self.__r_tree, self.__d_tree = self.__load_nodes()

        rprint(self.__r_tree)
        print(self.__d_tree.children[0].children[0].children[0], self.__d_tree.children[1])

    def queue_to_tree(self, lst: list) -> tuple[Tree, DNode]:
        """
        Transform a list of values into a tree graph
        """

        if lst[1]:
            raise Exception("The list not represents a valid tree")

        # transform list to queue struct
        queue = deque(lst)

        # init root path
        element = queue.popleft()

        r_root = Tree(element)  # root of rich tree
        d_root = DNode(element)  # root of DNode tree

        queue.popleft()

        # initialize dummy as root
        r_dummy = r_root
        d_dummy = d_root

        # initialize node queue
        r_node_queue = deque()
        d_node_queue = deque()

        while queue:
            # While queue is not empty keep loop.
            # This is done because we will be adding
            # nodes to the tree as long as there are
            # elements in the queue; if the queue is
            # already empty, we should not add
            # more nodes to the tree

            # Pop fist element from queue
            element = queue.popleft()

            # An empty element signifies a level jump
            # in the branches of the tree, so when an
            # empty element is detected, we need to
            # change the pointer (dummy) to the next
            # one in the node queue
            if not element:

                if len(r_node_queue) > 0:
                    # If the node list still has elements,
                    # we change the pointer (dummy) to
                    # the next one in the list and proceed
                    # to the next iteration. This ensures
                    # that we add the children to the correct node
                    r_dummy = r_node_queue.popleft()
                    d_dummy = d_node_queue.popleft()

                    continue

                # If the node list is already empty
                # we break the loop
                break

            # We create a node with the next value
            # in the queue, then we add that node
            # to the node queue to later add its
            # corresponding children. Finally,
            # we add the node as a child to
            # the current pointer
            r_child = Tree(element)
            d_child = DNode(element, node_type=DNodeTypes(element.lower().split("_")[0]))

            r_node_queue.append(r_child)
            d_node_queue.append(d_child)

            r_dummy.add(r_child)
            d_dummy.add(d_child)

        return r_root, d_root

    def __load_nodes(self) -> tuple[Tree, DNode]:
        """
        Traverse the folder tree of the current path and transform it into an N-Tree.
        """

        # adjacent list to convert into tree
        lst = []
        types_list = []

        # queue to save tree levels
        queue = deque()

        # add current path
        queue.append(self.__path)

        # if the paths queue is not empty
        while len(queue) > 0:
            # set current path as first item of paths queue
            path = queue.popleft()

            # list of directories in current path
            add_list = [dr.name for dr in os.scandir(path) if dr.is_dir()]
            # add None element to switch tree level
            lst += add_list + [None]
            # add level to paths queue
            queue += deque([path + f"{dr_name}/" for dr_name in add_list])

        # return tree
        return self.queue_to_tree(lst)


if __name__ == '__main__':
    DataCenter()
    pass
