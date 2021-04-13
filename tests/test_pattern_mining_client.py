import unittest
import numpy as np
import pandas as pd
from module.utilities import stringify_dataframe
from module.patterncount import SequenceMap
from module.mining import EnumeratedPattern
from module.pattern_mining_client import format_output

class TestPatternClient(unittest.TestCase):

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
    lag = 1
    num_of_readings = 9
    anomalous_windows = engine_data.index[engine_data.ncwindow == 1].tolist()
    enum_patterns_inst = EnumeratedPattern(
        anomalous_windows, num_of_readings, lag)

    # Enumerating patterns
    test_pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(
        test_pattern_str, pattern_occurences)

    dimensions = ['engrpm', 'brkpw', 'nox']
    pattern_df = engine_data[dimensions].iloc[1:3]
    pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(pattern_str, pattern_occurences)

    dimensions = ['nox']
    pattern_df = engine_data[dimensions].iloc[4:6]
    test_pattern_str_2 = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(
        test_pattern_str_2, pattern_occurences)

    dimensions = ['brkpw']
    pattern_df = engine_data[dimensions].iloc[2:4]
    pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(pattern_str, pattern_occurences)

    dimensions = ['nox', 'brkpw']
    pattern_df = engine_data[dimensions].iloc[7:9]
    pattern_str = stringify_dataframe(pattern_df)
    pattern_occurences = seqmap_inst.find_pattern_occurences(pattern_df)
    enum_patterns_inst.enumerate_pattern(pattern_str, pattern_occurences)

    def test_format_output_threshold(self):
        output_df = format_output(['engrpm', 'brkpw', 'nox'], self.enum_patterns_inst,
                                  self.pattern_length, metric='crossk',
                                  filter_type='threshold', threshold=2.5)

        expected_df = pd.DataFrame({'engrpm': ['1755 1076', '1755 1076', ' '],
                                    'brkpw': ['574 158', '574 158', '158 610'],
                                    'nox': [' ', '48 27', ' '],
                                    'Count': [3, 1, 1],
                                    'Support': [0.5555555555555556,
                                                0.2222222222222222,
                                                0.2222222222222222],
                                    'Kvalue': [2.5, 3.0, 3.0],
                                    'Confidence': [1.0, 1.0, 1.0],
                                    'Single Occurence Index': [1, 1, 2]})

        self.assertDictEqual(output_df.to_dict(), expected_df.to_dict())

    def test_format_output_topk(self):
        output_df = format_output(['engrpm', 'brkpw', 'nox'], self.enum_patterns_inst,
                                  self.pattern_length, metric='support',
                                  filter_type='topk', k=2)

        expected_df = pd.DataFrame({'engrpm': ['1755 1076', ' '],
                                    'brkpw': ['574 158', ' '],
                                    'nox': [' ', '48 26'],
                                    'Count': [3, 2],
                                    'Support': [0.5555555555555556,
                                                0.3333333333333333],
                                    'Kvalue': [2.5, 2.25],
                                    'Confidence': [1.0, 1.0],
                                    'Single Occurence Index': [1, 4]})

        self.assertDictEqual(output_df.to_dict(), expected_df.to_dict())


if __name__ == '__main__':
    unittest.main()
