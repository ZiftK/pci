from abc import ABC, abstractmethod

from rich.tree import Tree
from rich import print as rprint

from enum import Enum

from collections import deque

from pandas import DataFrame, read_csv

from PIL import Image

import os


class DNodeTypes(Enum):
    ROOT = "root"
    FUNCTION = "f"
    INPUT_SET = "is"
    OUTPUT_SET = "os"
    ANALYZE_SET = "as"
    PLOT = "p"

    @staticmethod
    def csv_types():
        return [
            DNodeTypes.INPUT_SET,
            DNodeTypes.OUTPUT_SET,
            DNodeTypes.ANALYZE_SET
        ]

    @staticmethod
    def image_types():
        return [
            DNodeTypes.PLOT
        ]


class DNode(ABC):

    def __init__(self, dir_name: str, node_type: DNodeTypes = DNodeTypes.ROOT, *args, **kwargs) -> None:
        """
        DNode is an N-Tree-like class designed to manage files and folders related to and generated by PCI tests
        Params
        ------
        :param dir_name: Directory name of the Node
        :param node_type: DNode type [root,function,input,output,analysis,plot]
        """

        # node type
        self.__type: DNodeTypes = node_type

        # clear dir name to get node name
        self.__name = dir_name.replace(self.__type.value + "_", "")

        # directory name
        self.__dir_name = dir_name

        # try to get build params
        self.__build_params: str = kwargs.get("build_params", "NA")

        # set default path as root path with directory name
        d_path = "/".join(__file__.split("\\")[:-1])
        self.__path = kwargs.get('path', f'{d_path}/{dir_name}')

        # node children
        self.children: list = []

        # node content
        self.content = None

    def add(self, child) -> None:
        """
        Adds a DNode child
        :param child: node child
        """
        child.path = f"{self.path}/{child.dir_name}"
        self.children.append(child)

    def load_content(self):
        """
        Load Node content from Node path
        """
        path_content = os.scandir(self.path)

        if not ("build_params.txt" in path_content):
            # TODO: Log error
            pass

        # TODO: save build params
        return path_content

    @abstractmethod
    def generate_content(self):
        """
        Generates content depending on the specific node implementation
        """

    @abstractmethod
    def save_content(self):
        """
        Save Node content
        """

    @property
    def name(self) -> str:
        """
        Returns the node name
        """
        return self.__name

    @property
    def node_type(self):
        return self.__type

    @property
    def dir_name(self):
        """
        Returns the directory name
        """
        return self.__dir_name

    @property
    def path(self) -> str:
        """
        Returns the directory path
        """
        return self.__path

    @path.setter
    def path(self, new_path: str) -> None:
        """
        Sets the directory path, only if new path is not empty
        """
        self.__path = new_path if new_path != "" else self.__path

    @property
    def build_params(self) -> str:
        """
        Returns the build params text
        """
        return self.__build_params

    @build_params.setter
    def build_params(self, value: str) -> None:
        """
        Sets the build params as value if value is not empty
        """
        self.__build_params = value if value != "" else "NA"

    def __str__(self):
        build_params = self.build_params.replace("\n", "\n\t")
        build_params = build_params.replace("=", " = ")
        return ("\n"
                f"Name: {self.__name}\n"
                f"Type: {self.__type.name}\n "
                f"Build:\n\t{build_params}\n"
                f"Path: {self.__path}"
                "\n")


class DRNode(DNode):

    def __init__(self, dir_name: str, node_type: DNodeTypes = DNodeTypes.ROOT, *args, **kwargs):
        super.__init__(dir_name, node_type, *args, **kwargs)

    def load_content(self):
        path_content = super().load_content()

    def generate_content(self):
        # TODO: add logic
        pass

    def save_content(self):
        # TODO: add logic
        pass


class DFNode(DNode):

    def __init__(self, dir_name: str, node_type: DNodeTypes = DNodeTypes.ROOT, *args, **kwargs):
        super.__init__(dir_name, node_type, *args, **kwargs)

    def load_content(self):
        path_content = super().load_content()

        if not ("content.io" in path_content):
            # TODO: Log error
            pass

        # TODO: save content

    def generate_content(self):
        # TODO: add logic
        pass

    def save_content(self):
        # TODO: add logic
        pass



def queue_to_tree(lst: list) -> tuple[Tree, DNode]:
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

        r_dummy.children.append(r_child)

        d_dummy.add(d_child)

    return r_root, d_root


class DataCenter:

    def __init__(self, *args, **kwargs) -> None:
        """
        Facade for managing test data for PCI
        """

        super().__init__(*args, **kwargs)

        # root path
        self.__path = kwargs.get('path', "\\".join(__file__.split("\\")[:-1]) + "\\")
        self.__p_splitter = kwargs.get('path_splitter', ".")
        # create NTree from directories
        self.__r_tree, self.__d_tree = self.__load_nodes()
        self.__load_content()

        self.__d_dummy = self.__d_tree
        self.__r_dummy = self.__r_tree

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
        return queue_to_tree(lst)

    def __load_content(self):
        """
        Load the content of the database into an N-Tree
        """

        # queue to transversal iterate
        queue = deque([self.__d_tree])

        while queue:  # while queue is not empty

            # get first node in queue
            dummy = queue.popleft()

            # get content of node directory
            content = [file.name for file in os.scandir(dummy.path) if file.is_file()]

            # if "build_params.txt" exists in directory
            if "build_params.txt" in content:
                # load content
                with open(f"{dummy.path}/build_params.txt", "r") as f:
                    dummy.build_params = f.read()

            # if node is a csv node type, and has a content, load csv content
            if (dummy.node_type in DNodeTypes.csv_types()) and "content.csv" in content:
                dummy.value = read_csv(f"{dummy.path}/content")

            # if node is an image node type and has a content, load image
            elif dummy.node_type in DNodeTypes.image_types() and "content.png" in content:
                dummy.value = Image.open(f"{self.__path}/content.png")

            # add dummy children to queue
            queue += deque(dummy.children)

    def __search_by_path(self, path: str):
        """
        Searches for a node in the tree graph whose
        path matches the specified one; if not found, it throws an exception
        :param path: node path
        :return: tuple with match rich node and data node
        """
        # split path using class splitter
        path = path.split(self.__p_splitter)

        # set reach Tree and DNode tree heads
        d_dummy = self.__d_tree
        r_dummy = self.__r_tree

        # iterate through split path
        for dr in path:

            # search directory in DNode children
            n_d_dummy = next((node for node in d_dummy.children if node.dir_name == dr), None)

            # if not exists child with specified name, rais exception
            if not n_d_dummy:
                raise Exception("No such directory")

            # set rich dummy to child with match index
            r_dummy = r_dummy.children[d_dummy.children.index(n_d_dummy)]
            # set data dummy to child with match index
            d_dummy = n_d_dummy

        return r_dummy, d_dummy  # return dummy's tuple

    def cd(self, path):

        self.__r_dummy, self.__d_dummy = self.__search_by_path(path)

    def show(self):
        rprint(self.__r_dummy, end="\r")
        print(self.__d_dummy, end="\r")


if __name__ == '__main__':
    a = DataCenter()
    # a.cd("f_tanx")
    a.show()

    pass
