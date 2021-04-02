import unittest
import numpy as np
import pandas as pd
from module.utilities import stringify_dataframe
from module.patterncount.stategraph import SequenceMap
from module.mining.candidatepattern import EnumeratedPattern

class TestEnumeratedPattern(unittest.TestCase):

    engrpm = np.array([2015, 1755, 1076, 2015, 1755, 1076, 2015, 1755, 1076])
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
    enum_patterns_inst = EnumeratedPattern(
        seqmap_inst, anomalous_windows, num_of_readings)

    # Enumerating pattern
    lag = 1
    enum_patterns_inst.enumerate_pattern(pattern_df, lag)

    def test_enumerate_pattern(self):
        prep_metrics = [self.enum_patterns_inst._support[-1],
                        self.enum_patterns_inst._confidence[-1],
                        self.enum_patterns_inst._crossk[-1]]
        expected_metrics = [0.5556, 1.0, 2.5]

        self.assertListEqual(prep_metrics, expected_metrics)

    def test_pattern_exists(self):
        pattern_str = stringify_dataframe(self.pattern_df)
        self.assertTrue(self.enum_patterns_inst.pattern_exists(pattern_str))

    def test_is_above_threshold(self):
        self.assertTrue(
            self.enum_patterns_inst.is_above_threshold(0, 'crossk', 2.2))


if __name__ == '__main__':
    unittest.main()
