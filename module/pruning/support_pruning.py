# Apriori algorithm, uses support of lower nodes to prune parent nodes and prevent enumeration
import copy
import pandas as pd

from ..utilities import stringify_dataframe
from .pruning_strategy import PruningStrategy
from ..patterncount import PatternCountStrategy
from ..mining import EnumeratedPattern, LatticeGraph

# Specific constant
prune_type = 'parents'
threshold_metric = 'support'

class SupportPruning(PruningStrategy):

    def __init__(self, num_of_dim: int, data: pd.DataFrame,
                 enum_pattern_inst: EnumeratedPattern,
                 pattern_count_inst: PatternCountStrategy,
                 threshold_value: float) -> None:

        self._data = data
        self._threshold_value = threshold_value
        self._enum_pattern_inst = enum_pattern_inst
        self._lattice_graph_inst = LatticeGraph(num_of_dim)
        self._pattern_count_inst = pattern_count_inst

    def _reconstruct_pattern_df(self, start_index: int, end_index: int,
                                column_indexes: list) -> pd.DataFrame:
        """Summary
            uses information provided as arguments to slice and index specific parts of dataframe
        """
        return self._data[self._data.columns[column_indexes]].iloc[start_index: end_index]

    def prune_patterns(self, start_index: int, end_index: int) -> list:
        """Summary
            finds pattern occurences in the data, some findings below:
            1. If a pattern is enumerated, means it's parents are also enumerated
            and thus, they should be pruned.
            2. If a pattern is below threshold, prune it's parents following apriori
            3. No need to store pruning information of a pattern as it already enumerated
        """
        # avoid creation of lattice graph, again and again
        clone_lattice_graph_inst = copy.deepcopy(self._lattice_graph_inst)

        nodes_not_enumerated = 0
        lattice_graph = clone_lattice_graph_inst.get_graph()

        for level, nodes_per_level in sorted(lattice_graph.items(), reverse=True):
            for node_dimensions, latticenode_inst in nodes_per_level.items():

                # a. check if node is pruned, if yes, move to next node
                if latticenode_inst.is_node_pruned():
                    nodes_not_enumerated += 1
                    continue

                # b. would need to reconstruct the pattern dataframe from indexes, to enumerate
                pattern_df = self._reconstruct_pattern_df(
                    start_index, end_index, list(node_dimensions))

                # c. check if pattern already enumerated, if not then enumerate
                pattern_str = stringify_dataframe(pattern_df)
                pattern_index = self._enum_pattern_inst.find_pattern(
                    pattern_str)

                if pattern_index == -1:
                    # finding pattern occurences
                    pattern_occurences = self._pattern_count_inst.find_pattern_occurences(
                        pattern_df)

                    # pattern enumeration (counting cooccurences and other metrics)
                    self._enum_pattern_inst.enumerate_pattern(
                        pattern_str, pattern_occurences)

                else:
                    # just to count node that does not need to be enumerated due to hashing
                    nodes_not_enumerated += 1

                # d. Now that pattern is enumerated, we have threshold information about it
                # two conditions for pruning parent patterns (hashing and apriori)
                if pattern_index != -1 or not self._enum_pattern_inst.is_above_threshold(pattern_index,
                                                                                         threshold_metric,
                                                                                         self._threshold_value):
                    # pruning all parents if node not above
                    clone_lattice_graph_inst.prune_nodes_recursively(
                        level, node_dimensions, prune_type)

        return nodes_not_enumerated
