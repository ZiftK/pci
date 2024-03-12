
from rich.tree import Tree


class DNode(Tree):

    def __init__(self, name: str, *args, **kwargs) -> None:

        super().__init__(name, *args, **kwargs)

        self.__path: str = kwargs.get('path', "")
        self.__build_params: dict = kwargs.get('build_params', dict())
class DataCenter:

    def __init__(self):

        pass


if __name__ == '__main__':


    pass