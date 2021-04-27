"""Summary
    Mines patterns using enumeration, pruning and pattern counting strategy
"""
from tqdm import tqdm
from bisect import bisect_left

from .utilities import print_fun
from .pruning import PruningStrategy
from .mining import EnumeratedPattern


class PatternMiner:
    def __init__(self, pattern_length: int, num_of_dims: int,
                 invalid_seq_indexes: list,
                 enum_pattern_inst: EnumeratedPattern,
                 pruning_inst: PruningStrategy):

        self._pattern_length = pattern_length
        self._num_of_dims = num_of_dims
        self._invalid_seq_indexes = invalid_seq_indexes
        self._enum_pattern_inst = enum_pattern_inst
        self._pruning_inst = pruning_inst

        self._lag = enum_pattern_inst.get_lag()
        self._num_of_readings = enum_pattern_inst.get_num_of_readings()
        self._anomalous_windows = enum_pattern_inst.get_anomalous_windows()

        self._visited_indexes = set()

    def mine(self) -> None:
        """Summary
            Mines sequence of co-occurence patterns
        """
        # O(n^3) for each window and each lag consider each combination of subsets
        saved_enumerations = 0
        valid_seq_count = 0

        for window_index in tqdm(self._anomalous_windows):
            for current_lag_val in range(self._lag, -1, -1):

                # formulating patterns by defining indexes
                start_pattern_index = window_index - current_lag_val
                end_pattern_index = start_pattern_index + self._pattern_length

                if self._is_valid_seq(start_pattern_index, end_pattern_index):
                    valid_seq_count += 1

                    # Adding indexes to visited
                    self._visited_indexes.add(
                        (start_pattern_index, end_pattern_index))

                    # Prune and enumerate patterns
                    saved_enumerations += self._pruning_inst.prune_and_enumerate_patterns(
                        start_pattern_index, end_pattern_index)

        # Calculating computations saved
        total_patterns = valid_seq_count * (2 ** self._num_of_dims - 1)
        message = 'Completed Mining, Pattern Enumerations saved: ({0} / {1})'
        print_fun(message.format(saved_enumerations, total_patterns))

    def _is_valid_seq(self, start_idx: int, end_idx: int) -> bool:
        """Summary
            Determines if current sequence is:
            1. Already visited
            2. Out of bounds
            3. Across any of violated indexes
        """
        # Most elegant and fast solution
        if start_idx < 0:
            return False

        if end_idx > self._num_of_readings:
            return False

        if (start_idx, end_idx) in self._visited_indexes:
            return False

        def BinarySearch(a: list, x) -> int:
            i = bisect_left(a, x)
            if i != len(a) and a[i] == x:
                return i
            else:
                return -1

        if len(self._invalid_seq_indexes) > 0:
            for i in range(start_idx + 1, end_idx):
                if BinarySearch(self._invalid_seq_indexes, i) != -1:
                    return False

        return True
