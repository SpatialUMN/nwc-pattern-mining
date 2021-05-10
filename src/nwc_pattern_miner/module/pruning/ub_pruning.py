# Apriori algorithm, uses support of lower nodes to prune parent nodes and prevent enumeration
import sys
import pandas as pd

from ..utilities import stringify_dataframe
from .pruning_strategy import PruningStrategy
from ..patterncount import PatternCountStrategy
from ..mining import EnumeratedPattern, LatticeGraph

# Specific constant
max_type = 'max'
min_type = 'min'
parent_prune_type = 'parents'
child_prune_type = 'children'
support_metric = 'support'
crossk_metric = 'crossk'

class UBPruning(PruningStrategy):

    def __init__(self, num_of_dims: int, data: pd.DataFrame,
                 enum_pattern_inst: EnumeratedPattern,
                 pattern_count_inst: PatternCountStrategy,
                 threshold_support_value: float,
                 threshold_crossk_value: float) -> None:

        self._data = data
        self._data_values = data.values
        self._num_of_dims = num_of_dims
        self._threshold_support_value = threshold_support_value
        self._threshold_crossk_value = threshold_crossk_value
        self._enum_pattern_inst = enum_pattern_inst
        self._crossk_const = enum_pattern_inst._crossk_const
        self._pattern_count_inst = pattern_count_inst
        self._singleton_joint_counts = dict()

    def _reconstruct_pattern_df(self, start_index: int, end_index: int,
                                column_indexes: list) -> pd.DataFrame:
        """Summary
            uses information provided as arguments to slice and index specific parts of dataframe
        """
        return pd.DataFrame(self._data_values[start_index:end_index, column_indexes],
                            columns=self._data.columns[column_indexes])

    def _apriori_pruning(self, start_index: int, end_index: int,
                         lattice_graph_inst: LatticeGraph, level: int) -> None:
        """Summary
            To prune patterns using support / apriori bottom up
        Args:
            start_index (int): start index of pattern
            end_index (int): end index of pattern
            level (int): Level in the lattice graph
        """
        nodes_enumerated = 0
        entire_level_pruned = True
        nodes_per_level = lattice_graph_inst.get_graph()[level]

        for node_dimensions, latticenode_inst in nodes_per_level.items():

            # a. check if node is pruned, if yes, move to next node
            if latticenode_inst.is_node_pruned():
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

                # Counting nodes enumerated
                nodes_enumerated += 1

            # Storing joint counts of singletons / leaves
            if len(node_dimensions) == 1:
                leaf_joinset_count = self._enum_pattern_inst._joinset_cardinality[pattern_index]
                self._singleton_joint_counts[node_dimensions[0]
                                             ] = leaf_joinset_count

            # d. Now that pattern is enumerated, we have threshold information about it
            # two conditions for pruning parent patterns (hashing and apriori)
            if not self._enum_pattern_inst.is_above_threshold(pattern_index,
                                                              support_metric,
                                                              self._threshold_support_value):
                # pruning all parents if node not above
                lattice_graph_inst.prune_nodes_recursively(
                    level, node_dimensions, parent_prune_type)
            else:
                entire_level_pruned = False

        return nodes_enumerated, entire_level_pruned

    def _get_joint_singleton_count(self, node_dimensions: set, count_type: str):
        """Summary
            get join cardinaltiy max / min count of leaves
        Args:
            node_dimensions (set): Node for which UBmax and UBmin is required
            count_type (str): max / min for type of joint singleton count
        """
        joint_count = -1
        if count_type == max_type:
            for key, val in self._singleton_joint_counts.items():
                if key in node_dimensions and val > joint_count:
                    joint_count = val
        else:
            # it's UBmin count
            joint_count = sys.maxint
            for key, val in self._singleton_joint_counts.items():
                if key in node_dimensions and val < joint_count:
                    joint_count = val

        return joint_count

    def _upperbound_pruning(self, start_index: int, end_index: int,
                            lattice_graph_inst: LatticeGraph, level: int) -> None:
        """Summary
            UB based pruning top down
        Args:
            start_index (int): Description
            end_index (int): Description
            lattice_graph_inst (LatticeGraph): Description
            level (int): Description
        """
        nodes_enumerated = 0
        stop_execution = False
        nodes_per_level = lattice_graph_inst.get_graph()[level]

        for node_dimensions, latticenode_inst in nodes_per_level.items():

            # a. check if node is pruned, if yes, move to next node
            if latticenode_inst.is_node_pruned():
                continue

            # b. check not root, then calculate UBmax and UBmin before pattern expansion
            if level != 0:
                max_leaf_count = self._get_joint_singleton_count(
                    node_dimensions, max_type)
                ub_max = self._crossk_const * \
                    (max_leaf_count / latticenode_inst.superpattern_count)

                if ub_max > 0 and ub_max < self._threshold_crossk_value:
                    # prune all children of current node
                    lattice_graph_inst.prune_nodes_recursively(
                        level, node_dimensions, parent_prune_type)
                    continue

            # c. else would need to reconstruct the pattern dataframe from indexes, to enumerate
            pattern_df = self._reconstruct_pattern_df(
                start_index, end_index, list(node_dimensions))

            # d. check if pattern already enumerated, if yes, then prune all children
            pattern_str = stringify_dataframe(pattern_df)
            pattern_index = self._enum_pattern_inst.find_pattern(
                pattern_str)
            if pattern_index != -1:
                # step different from apriori pruning as (children can have different parents,
                # but a parent would have exact same children)
                # prune all children of current node, without checking for threshold

                # If root, stop for the entire lattice
                if level == 0:
                    # entire lattice pruned
                    stop_execution = True
                    return nodes_enumerated, stop_execution

                # else just prune children of the current node
                lattice_graph_inst.prune_nodes_recursively(
                    level, node_dimensions, parent_prune_type)
                continue

            # till here means node either root, not enumerated or UBmax > threshold
            # e. Any of the above cases, would have to count and enumerate the pattern,
            # finding pattern occurences
            pattern_occurences = self._pattern_count_inst.find_pattern_occurences(
                pattern_df)
            # pattern enumeration (counting cooccurences and other metrics)
            self._enum_pattern_inst.enumerate_pattern(
                pattern_str, pattern_occurences)
            # Counting nodes enumerated
            nodes_enumerated += 1

            # f. update if pattern above support and crossk-threshold (return T/F)
            self._enum_pattern_inst.is_above_threshold(
                -1, crossk_metric, self._threshold_crossk_value)
            self._enum_pattern_inst.is_above_threshold(
                -1, support_metric, self._threshold_support_value)

            # g. obtain current pattern count
            curr_pattern_count = self._enum_pattern_inst._patterncount[-1]

            # h. If the node is root, check UBmax
            if level == 0:
                max_leaf_count = max(self._singleton_joint_counts.values())
                ub_max = self._crossk_const * \
                    (max_leaf_count / curr_pattern_count)

                if ub_max < self._threshold_crossk_value:
                    # entire lattice pruned
                    stop_execution = True
                    return nodes_enumerated, stop_execution

            # i. If continued till here, transfer the superpattern counts to the children
            bigger_count = curr_pattern_count
            if bigger_count < latticenode_inst.superpattern_count:
                bigger_count = latticenode_inst.superpattern_count

            for child_dimensions in latticenode_inst.get_children():
                latticechild_inst = lattice_graph_inst.get_node(
                    level + 1, child_dimensions)

                if latticechild_inst.superpattern_count < bigger_count:
                    latticechild_inst.superpattern_count = bigger_count

        return nodes_enumerated, stop_execution

    def prune_and_enumerate_patterns(self, start_index: int, end_index: int) -> list:
        """Summary
            finds pattern occurences in the data, some findings below:
            1. Enumerate all the leaves, just like apriori
            2. Enumerate root, if not already
            3. Use UB pruning while top-down traversal, propogate highest superpatterncount
            to the children
            4. Use Apriori to prune while bottom-up traversal
        """
        lattice_graph_inst = LatticeGraph(self._num_of_dims)

        topmost_level = 0
        stop_execution = False
        nodes_enumerated = 0
        total_nodes_enumerated = 0
        bottomost_level = self._num_of_dims - 1

        while topmost_level <= bottomost_level and not stop_execution:

            # bottom-up
            nodes_enumerated, stop_execution = self._apriori_pruning(
                start_index, end_index, lattice_graph_inst, bottomost_level)

            bottomost_level -= 1
            total_nodes_enumerated += nodes_enumerated

            # top-down
            if topmost_level <= bottomost_level and not stop_execution:
                nodes_enumerated, stop_execution = self._upperbound_pruning(
                    start_index, end_index, lattice_graph_inst, topmost_level)

                topmost_level += 1
                total_nodes_enumerated += nodes_enumerated

        return (lattice_graph_inst.num_of_nodes - total_nodes_enumerated)
