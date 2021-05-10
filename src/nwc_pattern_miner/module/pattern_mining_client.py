"""Summary
API end point to read input, delegate mining by creating strategies, and finally
formats patterns identified in a specific format
"""
import pandas as pd

from .utilities import print_fun
from .pruning import SupportPruning, UBPruning
from .mining import EnumeratedPattern
from .patterncount import SequenceMap
from .patternminer import PatternMiner

# Specific constants
dummy_col_name = 'dummy'
dummy_col_value = 'na'
ub_pruning = 'bi-dr'
supp_pruning = 'apriori'
support_output_metrics = ['crossk', 'support']
supported_output_types = ['threshold', 'topk']


def mine_sequence_patterns(series_df: pd.DataFrame, nc_window_col: str,
                           support_threshold: float, crossk_threshold: float,
                           pattern_length: int, confidence_threshold: float = -1,
                           lag: int = 0, invalid_seq_indexes: list = [],
                           output_metric: str = 'crossk',
                           output_type: str = 'topk',
                           output_threshold: float = -1, topk: int = 100,
                           pruning_type: str = 'apriori') -> pd.DataFrame:
    """Summary
        Main function / interface for the package (Driver function)
    """
    # Creating concrete strategy for counting patterns
    message = 'Counting pattern occurences'
    print_fun(message, status='step')
    seqmap_inst = SequenceMap(series_df, nc_window_col)
    seqmap_inst.init_seq_map(pattern_length)

    # Instantiating instance for pattern enumeration
    num_of_readings = len(series_df)
    anomalous_windows = series_df.index[series_df[nc_window_col] == 1].tolist()
    enum_patterns_inst = EnumeratedPattern(
        anomalous_windows, num_of_readings, lag)

    # Instantiate concrete strategy for pruning
    num_of_dims = series_df.shape[1] - 1
    if pruning_type == supp_pruning:
        pruning_inst = SupportPruning(
            num_of_dims, series_df, enum_patterns_inst, seqmap_inst, support_threshold)
    else:
        pruning_inst = UBPruning(num_of_dims, series_df, enum_patterns_inst,
                                 seqmap_inst, support_threshold, crossk_threshold)

    # Instantiate miner instance
    message = 'Processing Anomalous Windows'
    print_fun(message, status='step')

    patternminer_inst = PatternMiner(
        pattern_length, num_of_dims, invalid_seq_indexes, enum_patterns_inst, pruning_inst)
    patternminer_inst.mine()

    # Preparing final output
    patterns_mined_df = pd.DataFrame()
    col_names = series_df.columns[series_df.columns != nc_window_col].tolist()

    patterns_mined_df = format_output(col_names, enum_patterns_inst,
                                      pattern_length, output_metric, output_type,
                                      k=topk, threshold=output_threshold)

    return patterns_mined_df


def format_output(col_names: list, enum_pattern_inst: EnumeratedPattern,
                  pattern_length: int, metric: str, filter_type: str, k: int = -1,
                  threshold: float = -1) -> pd.DataFrame:
    """Summary
        Retuns patterns and associated metrics as a dataframe
    Args:
        pattern_length (int): Fixed pattern length
        metric (str): Type of metric to filter patterns upon
        filter_type (str): filter_type of filter (topk vs threshold based)
        k (int, optional): K if filter_type=topk
        threshold (float, optional): threshold value if filter_type=threshold
    """
    message = 'Formatting Enumerated Patterns ({0}) as Output via: ({1})'.format(
        enum_pattern_inst.num_of_patterns, filter_type)
    print_fun(message, status='step')

    pattern_indexes = enum_pattern_inst.get_pattern_indexes(
        metric, filter_type, k, threshold)

    patterns_list = enum_pattern_inst.get_patterns(pattern_indexes)
    pattern_metrics = enum_pattern_inst.get_pattern_metrics(pattern_indexes)

    all_patterns_df = convert_patterns_to_df(
        patterns_list, col_names, pattern_length)

    # Join patterns and their respective metrics, side by side
    return pd.concat([all_patterns_df, pattern_metrics], axis=1)


def convert_patterns_to_df(patterns_list: list, col_names: list,
                           pattern_length: int) -> pd.DataFrame:
    """Summary
        Convert list of jsons into a dataframe of patterns
    Args:
        col_names (list): list of column names for attributes
        patterns_list (list): list of all patterns enumerated, as jsons
        pattern_length (int): Individual pattern length is constant
    """

    all_patterns_df = pd.DataFrame()
    for pattern_str in patterns_list:

        # Converting from json to dataframe
        pattern_df = pd.read_json(pattern_str)

        # Convert all values to string
        pattern_df = pattern_df.applymap(str)

        # Insert empty attributes columns, in place, not present in current pattern
        for i, attr_col_name in enumerate(col_names):
            if attr_col_name not in pattern_df.columns:
                pattern_df.insert(i, attr_col_name, [''] * pattern_length)

        # Insert a dummy index columns for grouping
        pattern_df[dummy_col_name] = dummy_col_value

        # Compress multiple pattern rows into one row
        compress_pattern_df = pattern_df.groupby(dummy_col_name).apply(
            lambda x: [' '.join(x[i]) for i in col_names]).apply(pd.Series).reset_index(drop=True)
        compress_pattern_df.columns = col_names

        # append each pattern
        all_patterns_df = all_patterns_df.append(compress_pattern_df)

    return all_patterns_df.reset_index(drop=True)
