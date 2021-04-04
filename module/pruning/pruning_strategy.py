import pandas as pd
from abc import ABCMeta, abstractmethod

class PruningStrategy(metaclass=ABCMeta):

    @abstractmethod
    def prune_patterns(self, start_index: int, end_index: int) -> list:
        pass

    @abstractmethod
    def _reconstruct_pattern_df(self, start_index: int, end_index: int,
                                column_indexes: list) -> pd.DataFrame:
        pass
