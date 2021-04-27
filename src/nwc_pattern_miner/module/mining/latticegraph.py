# To represent all subsets in a hierarchical fashion
# A Dag; storing parent child relationships; Level order traversal
import json
import itertools
from .latticenode import LatticeNode

# File specific constants
prune_types = ['parents', 'children']

class LatticeGraph:

    def __init__(self, num_of_dim: int):
        # Instantiates atleast first level, stores actual nodes in
        # {depth: {dimension: latticenode_instance }} mappings
        self._num_of_dim = num_of_dim
        self._nodes_by_level = {0: dict()}
        self.num_of_nodes = 2 ** num_of_dim - 1

        # Initializing lattice graph
        self._init_graph(num_of_dim)

    def _create_node(self, depth: int, dimension: tuple) -> None:
        """Summary
            creates node at a certain depth, with a certain dimension
        """
        self._nodes_by_level[depth][dimension] = LatticeNode(dimension)

    def _init_graph(self, num_of_dim: int) -> None:
        """Summary
            creates all nodes and their interconnection in the graph
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
                    self._create_node(depth, parent)

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
                        self._create_node(depth + 1, child)

                    # Add parent information in the children
                    self.get_node(depth + 1, child).add_parents([parent])

            # Switch queues for next level
            level_queue = temp_queue

    def get_node(self, depth: int, dimension: tuple) -> LatticeNode:
        """Summary
            should be called after graph initialization
        """
        return self._nodes_by_level[depth][dimension]

    def get_graph(self) -> dict:
        """Summary
            method to expose lattice graph created
        """
        return self._nodes_by_level

    def prune_nodes_recursively(self, depth: int, dimension: tuple,
                                prune_type: str, num_of_pruned_nodes: int = 0) -> int:
        """Summary
            prunes nodes recursively both sides either parents or children
        Args:
            depth (int): depth at which node is located
            dimension (tuple): dimensions of the node
            prune_type (str): Parents or children
        """
        lattice_node_inst = self.get_node(depth, dimension)

        if lattice_node_inst.is_node_pruned():
            return num_of_pruned_nodes

        # Else we prune the node and move forward
        lattice_node_inst.prune_node()
        num_of_pruned_nodes += 1

        # deciding where to on the basis of prune type
        next_depth = 0
        next_level_nodes = list()
        if prune_type == prune_types[0]:
            next_depth = depth - 1
            next_level_nodes = lattice_node_inst.get_parents()

        elif prune_type == prune_types[1]:
            next_depth = depth + 1
            next_level_nodes = lattice_node_inst.get_children()

        else:
            raise Exception('Wrong prune type provided')

        # Using DFS for the task
        for next_level_dimension in next_level_nodes:
            num_of_pruned_nodes += self.prune_nodes_recursively(
                next_depth, next_level_dimension, prune_type, num_of_pruned_nodes)

        return num_of_pruned_nodes

    def _get_graph_as_dict(self):
        """Summary
            private method to convert graph to a recursive dictionary
        stringifying dimensions to beautify and print graphs as jsons (json does
        not allow tuples as keys)
        """
        temp_dict = dict()
        for depth, dimension_dict in self._nodes_by_level.items():
            temp_dict[depth] = dict()
            for dimension, latticenode in dimension_dict.items():
                if not latticenode.is_node_pruned():
                    temp_dict[depth][str(dimension)] = {
                        prune_types[0]: [str(x) for x in latticenode.get_parents()],
                        prune_types[1]: [str(x) for x in latticenode.get_children()]}

        return temp_dict

    def __str__(self):
        """Summary
            returns graph as a string, excluding the pruned nodes
        """
        return str(self._get_graph_as_dict())

    def beautify_print_graph(self) -> None:
        """Summary
            prints graph with proper identations
        """
        print(json.dumps(self._get_graph_as_dict(), indent=4))
