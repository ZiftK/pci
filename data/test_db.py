from rich.tree import Tree
from rich import print as rprint

from enum import Enum


class DNodeTypes(Enum):
    FUNCTION = 0
    INPUT_SET = 1
    OUTPUT_SET = 2
    ANALYZE_SET = 3
    GRAPHIC = 4


class DNode(Tree):

    def __init__(self, name: str, type: DNodeTypes, *args, **kwargs) -> None:
        super().__init__(name, *args, **kwargs)

        self.__path: str = kwargs.get('path', "")
        self.__type = DNodeTypes(type)
        self.__build_params: dict = kwargs.get('build_params', dict())

    def __str__(self):

        return f"Type: {self.__type.name}\n Path: {self.__path}\n Build: {self.__build_params}"


class DataCenter:

    def __init__(self):
        pass


if __name__ == '__main__':

    a = DNode(name="A", type=DNodeTypes.FUNCTION)
    rprint(a)
    print(a)

    pass
