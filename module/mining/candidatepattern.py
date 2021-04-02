"""Summary
Hashes / enumerates patterns, finds top patterns
"""
import pandas as pd
from ..patterncount import SequenceMap
from ..utilities import stringify_dataframe

class EnumeratedPattern:

    def __init__(self, seqmap_inst: SequenceMap, anomalous_windows: list, num_of_readings: int):
        """Summary
        patterns: to store already visited patterns (serialised dataframe segments)
        occurences: all positions of each dataframe segment
        features: dimensions involved in the dataframe segment
        support, confidence, crossk: support of the enumerated patterns
        anomalous_windows (list): occurences of anomalous windows
        """
        self._seqmap_inst = seqmap_inst
        self._num_of_readings = num_of_readings
        self._anomalous_windows = anomalous_windows
        self._crossk_const = num_of_readings / len(anomalous_windows)
        self._patterns = list()
        self._occurences = list()
        self._patterncount = list()
        self._joinset_cardinality = list()
        self._unique_joinset_cardinality = list()
        self._support = list()
        self._confidence = list()
        self._crossk = list()

    def enumerate_pattern(self, pattern_df: pd.DataFrame, lag: int) -> None:
        """Summary
        Onboards a patterns (counting all it's metrics)
        Args:
            pattern_df (pd.DataFrame): dataframe segment representing sequence
        """
        pattern_str = stringify_dataframe(pattern_df)

        # pattern already enumerated
        if self.pattern_exists(pattern_str):
            return

        # Add pattern to enumeration list
        self._patterns.append(pattern_str)

        # finding all pattern occurences (costly operation)
        pattern_occurences = self._seqmap_inst.find_pattern_occurences(
            pattern_df)

        self.fill_pattern_counts(pattern_occurences)

        self.fill_pattern_joinsets(pattern_occurences, lag)

        self.fill_pattern_metrics()

    def fill_pattern_counts(self, pattern_occurences: list) -> None:
        """Summary
        Fills all fields that are directly derivable for the pattern
        """
        self._occurences.append(pattern_occurences)
        self._patterncount.append(len(pattern_occurences))

    def fill_pattern_joinsets(self, pattern_occurences: list, lag: int) -> None:
        """Summary
        Fill joinset cardinalities of pattern occurence w.r.t anomalous windows
        """
        joinset_count = 0
        unique_joinset_count = 0

        # Finds cooccurences of pattern and anomalous windows
        for index in pattern_occurences:
            temp_list = list(range(index, index + lag + 1))
            num_of_intersections = len(
                set(temp_list).intersection(self._anomalous_windows))

            if num_of_intersections > 0:
                joinset_count += num_of_intersections
                unique_joinset_count += 1

        self._joinset_cardinality.append(joinset_count)
        self._unique_joinset_cardinality.append(unique_joinset_count)

    def fill_pattern_metrics(self) -> None:
        """Summary
        Calculates various metrics for the patterns, like support, confidence, crossk
        """
        # Picking up last element
        patterncount = self._patterncount[-1]
        joinset_count = self._joinset_cardinality[-1]
        unique_joinset_count = self._unique_joinset_cardinality[-1]

        self._confidence.append(round(unique_joinset_count / patterncount, 4))
        self._support.append(round(joinset_count / self._num_of_readings, 4))

        crossk_var = joinset_count / patterncount
        self._crossk.append(round(self._crossk_const * crossk_var, 4))

    def pattern_exists(self, pattern_str: str):
        return pattern_str in self._patterns

    def is_above_threshold(self, pattern_index: int, metric: str,
                           threshold: float) -> bool:
        metric_values = list()

        if metric == 'crossk':
            metric_values = self._crossk
        elif metric == 'confidence':
            metric_values = self._confidence
        elif metric == 'support':
            metric_values = self._support
        else:
            raise Exception('Invalid metric type provided')

        return metric_values[pattern_index] >= threshold

    def __str__(self):
        desc = 'Number of patterns: {0} | Metrics recorded for: {1}'

        return desc.format(len(self._patterns, len(self._support)))
