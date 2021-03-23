import unittest
from module.mining.latticegraph import LatticeGraph

class TestLatticeGraph(unittest.TestCase):

    def test_init_graph(self):
        lattice_graph_inst = LatticeGraph()

        num_of_dims = 3
        lattice_graph_inst.init_graph(num_of_dims)

        expected_graph = ("{0: {'(0, 1, 2)': {'parents': [], "
                          "'children': ['(0, 1)', '(0, 2)', '(1, 2)']}}, "
                          "1: {'(0, 1)': {'parents': ['(0, 1, 2)'], "
                          "'children': ['(0,)', '(1,)']}, '(0, 2)': "
                          "{'parents': ['(0, 1, 2)'], 'children': "
                          "['(0,)', '(2,)']}, '(1, 2)': {'parents': ['(0, 1, 2)'], "
                          "'children': ['(1,)', '(2,)']}}, 2: {'(0,)': "
                          "{'parents': ['(0, 1)', '(0, 2)'], "
                          "'children': []}, '(1,)': {'parents': "
                          "['(0, 1)', '(1, 2)'], 'children': []}, '(2,)': "
                          "{'parents': ['(0, 2)', '(1, 2)'], 'children': []}}}")

        self.assertEqual(lattice_graph_inst.__str__(), expected_graph)

    def test_prune_children(self):
        lattice_graph_inst = LatticeGraph()

        num_of_dims = 3
        lattice_graph_inst.init_graph(num_of_dims)

        lattice_graph_inst.prune_nodes_recursively(1, (0, 1), 'children')

        expected_graph = ("{0: {'(0, 1, 2)': {'parents': [], "
                          "'children': ['(0, 1)', '(0, 2)', '(1, 2)']}}, "
                          "1: {'(0, 2)': "
                          "{'parents': ['(0, 1, 2)'], 'children': "
                          "['(0,)', '(2,)']}, '(1, 2)': {'parents': ['(0, 1, 2)'], "
                          "'children': ['(1,)', '(2,)']}}, 2: {'(2,)': "
                          "{'parents': ['(0, 2)', '(1, 2)'], 'children': []}}}")

        self.assertEqual(lattice_graph_inst.__str__(), expected_graph)

    def test_prune_parents(self):
        lattice_graph_inst = LatticeGraph()

        num_of_dims = 3
        lattice_graph_inst.init_graph(num_of_dims)

        lattice_graph_inst.prune_nodes_recursively(2, (0,), 'parents')

        expected_graph = ("{0: {}, 1: {'(1, 2)': {'parents': ['(0, 1, 2)'], "
                          "'children': ['(1,)', '(2,)']}}, 2: {'(1,)': "
                          "{'parents': ['(0, 1)', '(1, 2)'], 'children': []}, '(2,)': "
                          "{'parents': ['(0, 2)', '(1, 2)'], 'children': []}}}")

        self.assertEqual(lattice_graph_inst.__str__(), expected_graph)


if __name__ == '__main__':
    unittest.main()
