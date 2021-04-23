import unittest
import numpy as np
import pandas as pd
from module.patterncount import SequenceMap

class TestStateGraph(unittest.TestCase):

    engrpm = np.array([2015, 1755, 1076, 2015, 1755, 1076])
    brkpw = np.array([660, 574, 158, 610, 574, 158])
    nox = np.array([82, 48, 27, 13, 48, 26])
    ncwindow = np.array([0, 1, 0, 1, 0, 0])
    engine_data = pd.DataFrame(
        {'engrpm': engrpm, 'brkpw': brkpw, 'nox': nox, 'ncwindow': ncwindow})

    # Instantiating common instance
    pattern_length = 2
    seqmap_inst = SequenceMap(engine_data, 'ncwindow')
    seqmap_inst.init_seq_map(pattern_length)

    def test_init_seq_map(self):
        expected_seq_map = {'engrpm': {
            (2015, 1755): [0, 3],
            (1755, 1076): [1, 4],
            (1076, 2015): [2]},
            'brkpw': {(660, 574): [0],
                      (574, 158): [1, 4],
                      (158, 610): [2],
                      (610, 574): [3]},
            'nox': {(82, 48): [0],
                    (48, 27): [1],
                    (27, 13): [2],
                    (13, 48): [3],
                    (48, 26): [4]}}

        self.assertDictEqual(expected_seq_map, self.seqmap_inst._seq_hashmap)

    def test_find_pattern_occurences(self):
        dimensions = ['engrpm', 'brkpw', 'nox']
        found_occurences = self.seqmap_inst.find_pattern_occurences(
            self.engine_data[dimensions].iloc[1:3])
        expected_occurences = [1]

        self.assertListEqual(expected_occurences, found_occurences)


if __name__ == '__main__':
    unittest.main()
