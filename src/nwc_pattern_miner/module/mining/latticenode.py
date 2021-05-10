# Nodes in lattice graph and associated modules

class LatticeNode:
    def __init__(self, dimensions: tuple):
        self._parents = list()
        self._children = list()
        self._is_pruned = False
        self.superpattern_count = -1
        self.dimensions = dimensions

    def is_node_pruned(self):
        return self._is_pruned

    def prune_node(self):
        self._is_pruned = True

    def add_parents(self, parents: list) -> None:
        """Args:
            parents (list): list of lattice node instances
        """
        self._parents.extend(parents)

    def get_parents(self) -> list:
        return self._parents

    def add_children(self, children: list) -> None:
        """Args:
            children (list): list of lattice node instances
        """
        self._children.extend(children)

    def get_children(self) -> list:
        return self._children
