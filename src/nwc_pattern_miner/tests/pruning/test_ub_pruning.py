import unittest
import numpy as np
import pandas as pd
from module.patterncount import SequenceMap
from module.mining import EnumeratedPattern
from module.pruning import UBPruning

class TestUBPruning(unittest.TestCase):

    engrpm = np.array([2015, 1755, 1076, 2015, 1755, 1076, 2014, 1755, 1076])
    brkpw = np.array([660, 574, 157, 610, 574, 158, 660, 574, 158])
    nox = np.array([82, 48, 27, 13, 48, 26, 13, 48, 26])
    absp = np.array([82, 48, 27, 13, 48, 26, 13, 48, 26])
    ncwindow = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1])
    engine_data = pd.DataFrame(
        {'engrpm': engrpm, 'brkpw': brkpw, 'nox': nox, 'absp': absp, 'ncwindow': ncwindow})

    # Instantiating common instance
    pattern_length = 2
    seqmap_inst = SequenceMap(engine_data, 'ncwindow')
    seqmap_inst.init_seq_map(pattern_length)

    # Instantiating Enumeration of patterns
    lag = 1
    num_of_readings = 9
    anomalous_windows = engine_data.index[engine_data.ncwindow == 1].tolist()
    enum_patterns_inst = EnumeratedPattern(
        anomalous_windows, num_of_readings, lag)

    # Creating instance for pruning
    num_of_dims = 4
    threshold_support_value = 0.05
    threshold_crossk_value = 3.5
    pruning_inst = UBPruning(
        num_of_dims, engine_data, enum_patterns_inst, seqmap_inst,
        threshold_support_value, threshold_crossk_value)

    def test_prune_and_enumerate_patterns(self):
        # tested all code branches through variations of manual tests
        output_pruning = list()
        expected_pruning = [4, 15]

        # save by apriori and hashing
        start_index, end_index = (4, 6)
        output_pruning.append(
            self.pruning_inst.prune_and_enumerate_patterns(start_index, end_index))

        # save by apriori and hashing
        start_index, end_index = (7, 9)
        output_pruning.append(
            self.pruning_inst.prune_and_enumerate_patterns(start_index, end_index))

        self.assertListEqual(output_pruning, expected_pruning)


if __name__ == '__main__':
    unittest.main()
