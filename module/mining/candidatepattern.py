"""Summary
Hashes / enumerates patterns, finds top patterns
"""
import sys
import numpy as np
import pandas as pd
from ..utilities import print_fun

# Specific constants
patterns_alert_threshold = 1000
metric_names = ['support', 'crossk', 'confidence']
metric_col_names = ['Count', 'Support', 'Kvalue',
                    'Confidence', 'Single Occurence Index']

class EnumeratedPattern:

    def __init__(self, anomalous_windows: list, num_of_readings: int,
                 lag: int) -> None:
        """Summary
        patterns: to store already visited patterns (serialised dataframe segments)
        occurences: all positions of each pattern sequence
        features: dimensions involved in the dataframe segment
        support, confidence, crossk: support of the enumerated patterns
        anomalous_windows (list): occurences of anomalous windows
        """
        self._lag = lag
        self._num_of_readings = num_of_readings
        self._anomalous_windows = anomalous_windows
        self._crossk_const = num_of_readings / len(anomalous_windows)

        self._num_of_patterns = 0
        self._patterns = list()
        self._occurences = list()
        self._patterncount = list()
        self._joinset_cardinality = list()
        self._unique_joinset_cardinality = list()
        self._support = list()
        self._confidence = list()
        self._crossk = list()

    def enumerate_pattern(self, pattern_str: str, pattern_occurences: int) -> None:
        """Summary
            Onboards a patterns (counting all it's metrics)
        Args:
            pattern_str (str): dataframe segment representing sequence, as a string
        """
        # pattern already enumerated
        if self.find_pattern(pattern_str) != -1:
            return

        # Add pattern to enumeration list
        self._patterns.append(pattern_str)

        self._fill_pattern_counts(pattern_occurences)

        self._fill_pattern_joinsets(pattern_occurences)

        self._fill_pattern_metrics()

        # Monitoring memory size of instance
        self._num_of_patterns += 1
        if self._num_of_patterns % patterns_alert_threshold == 0:
            print_fun(self)

    def _fill_pattern_counts(self, pattern_occurences: list) -> None:
        """Summary
            Fills all fields that are directly derivable for the pattern
        """
        self._occurences.append(pattern_occurences)
        self._patterncount.append(len(pattern_occurences))

    def _fill_pattern_joinsets(self, pattern_occurences: list) -> None:
        """Summary
            Fill joinset cardinalities of pattern occurence w.r.t anomalous windows
        """
        joinset_count = 0
        unique_joinset_count = 0

        # Finds cooccurences of pattern and anomalous windows
        for index in pattern_occurences:
            temp_list = list(range(index, index + self._lag + 1))
            num_of_intersections = len(
                set(temp_list).intersection(self._anomalous_windows))

            if num_of_intersections > 0:
                joinset_count += num_of_intersections
                unique_joinset_count += 1

        self._joinset_cardinality.append(joinset_count)
        self._unique_joinset_cardinality.append(unique_joinset_count)

    def _fill_pattern_metrics(self) -> None:
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

    def _get_metric_values(self, metric: str) -> list:
        """Summary
            returns metric values depending upon the metric selected
        """
        metric_values = list()

        if metric == 'crossk':
            metric_values = self._crossk
        elif metric == 'confidence':
            metric_values = self._confidence
        elif metric == 'support':
            metric_values = self._support
        else:
            raise Exception('Invalid metric type provided')

        return metric_values

    def is_above_threshold(self, pattern_index: int, threshold_metric: str,
                           threshold_value: float) -> bool:
        """Summary
            returns pattern at a specific index to be above threshold or not
        """
        metric_values = self._get_metric_values(threshold_metric)

        return metric_values[pattern_index] >= threshold_value

    def find_pattern(self, pattern_str: str) -> int:
        """Summary
            returns index of enumerated pattern
        """
        if pattern_str in self._patterns:
            return self._patterns.index(pattern_str)

        return -1

    def get_pattern_indexes(self, metric: str, filter_type: str, k: int = -1,
                            threshold: float = -1) -> list:
        """Summary
            returns indexes of topK patterns or patterns above a threshold, depending upon metric
        """
        pattern_indexes = list()

        metric_values = self._get_metric_values(metric)
        num_of_patterns = len(metric_values)

        if filter_type == 'topk' and k != -1:
            pattern_indexes = sorted(
                range(num_of_patterns), key=lambda i: metric_values[i], reverse=True)[:k]

        elif filter_type == 'threshold' and threshold != -1:
            pattern_indexes = [i for i in range(
                num_of_patterns) if metric_values[i] >= threshold]

        else:
            raise Exception(
                'Insufficient / wrong parameters for finding pattern indexes')

        return pattern_indexes

    def get_patterns(self, pattern_indexes: list) -> list:
        """Summary
            returns list of patterns as strings, through list of indexes provided
        """
        return [self._patterns[i] for i in range(self._num_of_patterns) if i in pattern_indexes]

    def get_pattern_metrics(self, pattern_indexes: list) -> pd.DataFrame:
        """Summary
            Provided list of pattern indexes, returns five defined metrics of patterns
        in a structure format (only returning index of the first occurence)
        """
        # manual indexing due to length of pattern occurences being irregular
        pattern_occurences = [x[0] for i, x in enumerate(
            self._occurences) if i in pattern_indexes]
        return pd.DataFrame({metric_col_names[0]: np.array(self._patterncount)[pattern_indexes],
                             metric_col_names[1]: np.array(self._support)[pattern_indexes],
                             metric_col_names[2]: np.array(self._crossk)[pattern_indexes],
                             metric_col_names[3]: np.array(self._confidence)[pattern_indexes],
                             metric_col_names[4]: pattern_occurences})

    def __str__(self):
        size_of_inst = sys.getsizeof(self) / 1000000
        desc = 'Number of patterns enumerated: {0} | Memory size: {1} MB'

        return desc.format(len(self._num_of_patterns, size_of_inst))
