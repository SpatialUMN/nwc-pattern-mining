import pandas as pd
from abc import ABCMeta, abstractmethod

class PruningStrategy(metaclass=ABCMeta):

    @abstractmethod
    def prune_and_enumerate_patterns(self, start_index: int, end_index: int) -> list:
        pass

    def _reconstruct_pattern_df(self, start_index: int, end_index: int,
                                column_indexes: list) -> pd.DataFrame:
        """Summary
            uses information provided as arguments to slice and index specific parts of dataframe
        """
        return self._data[self._data.columns[column_indexes]].iloc[start_index: end_index]
