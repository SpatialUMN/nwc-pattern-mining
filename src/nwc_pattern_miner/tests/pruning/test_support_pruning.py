import unittest
import numpy as np
import pandas as pd
from module.patterncount import SequenceMap
from module.mining import EnumeratedPattern
from module.pruning import SupportPruning

class TestSupportPruning(unittest.TestCase):

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

    # Instantiating Enumeration of patterns
    lag = 1
    num_of_readings = 9
    anomalous_windows = engine_data.index[engine_data.ncwindow == 1].tolist()
    enum_patterns_inst = EnumeratedPattern(
        anomalous_windows, num_of_readings, lag)

    # Creating instance for pruning
    num_of_dims = 3
    threshold_value = 0.5
    pruning_inst = SupportPruning(
        num_of_dims, engine_data, enum_patterns_inst, seqmap_inst, threshold_value)

    def test_prune_and_enumerate_patterns(self):
        output_pruning = list()

        # save by apriori
        start_index, end_index = (1, 3)
        output_pruning.append(
            self.pruning_inst.prune_and_enumerate_patterns(start_index, end_index))

        # save by apriori and hashing
        start_index, end_index = (4, 6)
        output_pruning.append(
            self.pruning_inst.prune_and_enumerate_patterns(start_index, end_index))

        expected_pruning = [3, 6]

        self.assertListEqual(output_pruning, expected_pruning)


if __name__ == '__main__':
    unittest.main()
