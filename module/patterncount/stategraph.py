# Maintains a hashmap with single transitions for each attribute
import sys
import pandas as pd

from ..utilities import print_fun
from .count_strategy import PatternCountStrategy

class SequenceMap(PatternCountStrategy):

    def __init__(self, data: pd.DataFrame, nc_window_col_name: str):
        self._data = data
        self._data_length = data.shape[0]
        self._feature_col_names = list(
            data.columns[data.columns != nc_window_col_name])
        self._seq_hashmap = dict()

    def init_seq_map(self, pattern_length: int) -> None:
        """Summary
            Creates hashmap of sequences for efficient support count
        """
        # Separately for each dimension
        for fc in self._feature_col_names:
            if fc not in self._seq_hashmap:
                self._seq_hashmap[fc] = dict()

            # hashing complete pattern for each dimention
            for idx in range(self._data_length - pattern_length + 1):
                seq_key = tuple(self._data[fc].iloc[idx: idx + pattern_length])

                if seq_key not in self._seq_hashmap[fc]:
                    self._seq_hashmap[fc][seq_key] = list()

                self._seq_hashmap[fc][seq_key].append(idx)

        size_of_map = sys.getsizeof(self._seq_hashmap) / 1000000
        message = 'Completed Sequence Hashing for Support Count (HashSize): ' + str(
            size_of_map) + ' MB'
        print_fun(message)

    def find_pattern_occurences(self, pattern_df: pd.DataFrame) -> list:
        """Summary
            Finds pattern occurences in the data
        """

        combined_dim_occurences = list()
        for name in pattern_df.columns:
            dim_pattern = tuple(pattern_df[name])

            if dim_pattern not in self._seq_hashmap[name]:
                return list()

            combined_dim_occurences.append(
                self._seq_hashmap[name][dim_pattern])

        # finding intersection of all dimensions to report back complete pattern occurence
        return list(set(combined_dim_occurences[0]).intersection(*combined_dim_occurences))
