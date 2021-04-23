import unittest
import numpy as np
import pandas as pd

from module.pruning import SupportPruning
from module.patterncount import SequenceMap
from module.mining import EnumeratedPattern
from module.patternminer import PatternMiner

class TestPatternClient(unittest.TestCase):

    engrpm = np.array([2015, 1755, 1076, 2015, 1755, 1076, 2014, 1755, 1076])
    brkpw = np.array([660, 574, 158, 610, 574, 158, 660, 574, 158])
    nox = np.array([82, 48, 27, 13, 48, 26, 13, 48, 26])
    ncwindow = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1])
    engine_data = pd.DataFrame(
        {'engrpm': engrpm, 'brkpw': brkpw, 'nox': nox, 'ncwindow': ncwindow})

    # Instantiating pattern count instance
    pattern_length = 2
    seqmap_inst = SequenceMap(engine_data, 'ncwindow')
    seqmap_inst.init_seq_map(pattern_length)

    # Instantiating Enumeration of patterns
    lag = 1
    num_of_readings = 9
    anomalous_windows = engine_data.index[engine_data.ncwindow == 1].tolist()
    enum_patterns_inst = EnumeratedPattern(
        anomalous_windows, num_of_readings, lag)

    # Instantiate pruning instance
    num_of_dims = 3
    threshold_value = 0.5
    pruning_inst = SupportPruning(
        num_of_dims, engine_data, enum_patterns_inst, seqmap_inst, threshold_value)

    # Instantiate miner instance (sequence breaks between 4-5 and 6-7)
    invalid_seq_indexes = [5, 7]
    patternminer_inst = PatternMiner(
        pattern_length, num_of_dims, invalid_seq_indexes, enum_patterns_inst, pruning_inst)

    def test_mine(self):
        self.patternminer_inst.mine()

        output_pattern_count = self.enum_patterns_inst._patterncount

        expected_pattern_count = [2, 2, 1, 3,
                                  3, 1, 3, 1, 1, 1, 1, 2, 1, 1, 1, 2]

        self.assertListEqual(expected_pattern_count, output_pattern_count)


if __name__ == '__main__':
    unittest.main()
