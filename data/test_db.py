from rich.tree import Tree
from rich import print as rprint

from enum import Enum

from collections import deque

import os


def queue_to_tree(lst: list) -> Tree:
    """
    Transform a list of values into a tree graph
    """
    # init root path
    root = Tree("root")

    # initialize dummy as root
    dummy = root

    # transform list to queue struct
    queue = deque(lst)
    # initialize node queue
    node_queue = deque()

    while len(queue) > 0:
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

            if len(node_queue) > 0:
                # If the node list still has elements,
                # we change the pointer (dummy) to
                # the next one in the list and proceed
                # to the next iteration. This ensures
                # that we add the children to the correct node
                dummy = node_queue.popleft()
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
        child = Tree(element)
        node_queue.append(child)
        dummy.add(child)

    return root


class DNodeTypes(Enum):
    ROOT = "root"
    FUNCTION = "function"
    INPUT_SET = "input set"
    OUTPUT_SET = "output set"
    ANALYZE_SET = "analyze set"
    PLOT = "plot"


class DNode(Tree):

    def __init__(self, name: str, dtype: DNodeTypes, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

        self.__type = DNodeTypes(dtype)
        self.__build_params: dict = kwargs.get('build_params', dict())

    def __str__(self):
        return f"Type: {self.__type.value}\n Build: {self.__build_params}"


class DataCenter:

    def __init__(self, *args, **kwargs) -> None:
        self.__path = kwargs.get('path', "./")
        self.__load_nodes()

    def __load_nodes(self) -> None:
        # create root node
        root = DNode("root")
        self.__path += "root/"

        # scan directory
        content = os.scandir(self.__path)

        values = deque()

        while content:
            values.append([dr.name for dr in content if dr.is_dir()])

        pass


if __name__ == '__main__':
    tree_list = [
        "a", "b", "c", None,
        "a1", "a2", None,
        None,
        "c1", "c2", None,
        None,
        None,
        "c1.1"
    ]

    pass
