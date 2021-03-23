# To represent all subsets in a hierarchical fashion
# A Dag; storing parent child relationships; Level order traversal
import json
import itertools
from .latticenode import LatticeNode

class LatticeGraph:

    def __init__(self):
        # Initialises atleast first level
        self._nodes_by_level = {0: dict()}

    def get_node(self, depth: int, dimension: tuple) -> LatticeNode:
        """Summary
        Should be called after graph initialization
        """
        return self._nodes_by_level[depth][dimension]

    def create_node(self, depth: int, dimension: tuple) -> None:
        """Summary
        Creates node at a certain depth, with a certain dimension
        """
        self._nodes_by_level[depth][dimension] = LatticeNode(dimension)

    def init_graph(self, num_of_dim: int) -> None:
        """Summary
        Creates all nodes and their interconnection in the graph
        O(n^3): one loop for all levels; one for current level to update children
        ; and one loop for next level to update the parent
        Args:
            num_of_dim (int): Creates all subsets of number of dimension
        """
        level_queue = [tuple(range(num_of_dim))]

        # loop for all levels
        for n in range(num_of_dim - 1, 0, -1):
            temp_queue = list()
            depth = (num_of_dim - n - 1)

            # loop for current level to add childrens
            for parent in level_queue:

                # If node not already created by another superset
                if parent not in self._nodes_by_level[depth]:
                    self.create_node(depth, parent)

                # Getting all children combinations for parent
                children = list(itertools.combinations(parent, n))

                # Adding all children for next level
                temp_queue.extend(children)

                # Adding all children information in parent node
                self.get_node(depth, parent).add_children(children)

                # loop for next level to add parents (one by one for each child)
                for child in children:

                    # Create an entry for new level
                    if depth + 1 not in self._nodes_by_level:
                        self._nodes_by_level[depth + 1] = dict()

                    # if child already created by another superset
                    if child not in self._nodes_by_level[depth + 1]:
                        self.create_node(depth + 1, child)

                    # Add parent information in the children
                    self.get_node(depth + 1, child).add_parents([parent])

            # Switch queues for next level
            level_queue = temp_queue

    def __str__(self):
        """Summary
        To represent and test graph created
        """
        temp_dict = dict()
        for depth, dimension_dict in self._nodes_by_level.items():
            temp_dict[depth] = dict()
            for dimension, latticenode in dimension_dict.items():
                temp_dict[depth][dimension] = dict()
                temp_dict[depth][dimension]['parents'] = latticenode._parents
                temp_dict[depth][dimension]['children'] = latticenode._children

        return str(temp_dict)
