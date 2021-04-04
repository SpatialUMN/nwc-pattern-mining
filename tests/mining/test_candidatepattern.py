import unittest
import numpy as np
import pandas as pd
from module.utilities import stringify_dataframe
from module.patterncount.stategraph import SequenceMap
from module.mining.candidatepattern import EnumeratedPattern

class TestEnumeratedPattern(unittest.TestCase):

    engrpm = np.array([2015, 1755, 1076, 2015, 1755, 1076, 2014, 1755, 1076])
    brkpw = np.array([660, 574, 158, 610, 574, 158, 660, 574, 158])
    nox = np.array([82, 48, 27, 13, 48, 26, 13, 48, 26])
    ncwindow = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1])
    engine_data = pd.DataFrame(
        {'engrpm': engrpm, 'brkpw': brkpw, 'nox': nox, 'ncwindow': ncwindow})

    # Instantiating common instance
    pattern_length = 2
    seqmap_inst = SequenceMap(engine_data, 'ncwindow')
    seqmap_inst.init_seq_map(pattern_length)

    # Enumerating a pattern
    dimensions = ['engrpm', 'brkpw']
    pattern_df = engine_data[dimensions].iloc[1:3]

    # Instantiating Enumeration of patterns
    num_of_readings = 9
    anomalous_windows = engine_data.index[engine_data.ncwindow == 1].tolist()
    enum_patterns_inst = EnumeratedPattern(anomalous_windows, num_of_readings)

    # Enumerating patterns
    lag = 1
    test_pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(
        test_pattern_str, pattern_occurences, lag)

    dimensions = ['engrpm', 'brkpw', 'nox']
    pattern_df = engine_data[dimensions].iloc[1:3]
    pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(pattern_str, pattern_occurences, lag)

    dimensions = ['nox']
    pattern_df = engine_data[dimensions].iloc[4:6]
    test_pattern_str_2 = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(
        test_pattern_str_2, pattern_occurences, lag)

    dimensions = ['brkpw']
    pattern_df = engine_data[dimensions].iloc[2:4]
    pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(pattern_str, pattern_occurences, lag)

    dimensions = ['nox', 'brkpw']
    pattern_df = engine_data[dimensions].iloc[7:9]
    pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(pattern_str, pattern_occurences, lag)

    def test_enumerate_pattern(self):
        prep_metrics = [self.enum_patterns_inst._support[0],
                        self.enum_patterns_inst._confidence[0],
                        self.enum_patterns_inst._crossk[0]]
        expected_metrics = [0.5556, 1.0, 2.5]

        self.assertListEqual(prep_metrics, expected_metrics)

    def test_pattern_exists(self):
        self.assertEqual(
            self.enum_patterns_inst.find_pattern(self.test_pattern_str), 0)

    def test_get_pattern_indexes_topk(self):
        prep_pattern_indexes = self.enum_patterns_inst.get_pattern_indexes(
            metric='support', kind='topk', k=1)
        self.assertListEqual(prep_pattern_indexes, [0])

    def test_get_pattern_indexes_threshold(self):
        prep_pattern_indexes = self.enum_patterns_inst.get_pattern_indexes(
            metric='crossk', kind='threshold', threshold=2.7)
        self.assertListEqual(prep_pattern_indexes, [1, 3])

    def test_get_pattern_metrics(self):
        prep_dict = self.enum_patterns_inst.get_pattern_metrics(
            [0, 2, 4]).to_dict()
        expected_dict = pd.DataFrame({'Count': [3, 2, 2], 'Support': [0.5556, 0.3333, 0.3333], 'Kvalue': [
                                     2.5, 2.25, 2.25], 'Confidence': [1.0, 1.0, 1.0],
            'Single Occurence Index': [1, 4, 4]}).to_dict()
        self.assertDictEqual(prep_dict, expected_dict)

    def test_get_patterns(self):
        prep_patterns = self.enum_patterns_inst.get_patterns([0, 2])
        self.assertListEqual(
            prep_patterns, [self.test_pattern_str, self.test_pattern_str_2])

    def test_is_above_threshold(self):
        self.assertTrue(
            self.enum_patterns_inst.is_above_threshold(0, 'crossk', 2.2))


if __name__ == '__main__':
    unittest.main()
