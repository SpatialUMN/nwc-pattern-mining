from abc import ABCMeta, abstractmethod
import pandas as pd

class PatternCountStrategy(metaclass=ABCMeta):

    @abstractmethod
    def find_pattern_occurences(self, pattern_df: pd.DataFrame) -> list:
        pass
