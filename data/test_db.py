from rich.tree import Tree
from rich import print as rprint

from enum import Enum

import os


class DNodeTypes(Enum):
    FUNCTION = 0
    INPUT_SET = 1
    OUTPUT_SET = 2
    ANALYZE_SET = 3
    GRAPHIC = 4


class DNode(Tree):

    def __init__(self, name: str, dtype: DNodeTypes, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

        self.__type = DNodeTypes(dtype)
        self.__build_params: dict = kwargs.get('build_params', dict())

    def __str__(self):

        return f"Type: {self.__type.name}\n Build: {self.__build_params}"


class DataCenter:

    def __init__(self):
        pass


if __name__ == '__main__':

    a = DNode(name="A", dtype=DNodeTypes.FUNCTION)
    rprint(a)
    print(a)

    pass
