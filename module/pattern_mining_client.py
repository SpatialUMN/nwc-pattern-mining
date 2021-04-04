"""Summary
API end point to read input, delegate mining by creating strategies, and finally
formats patterns identified in a specific format
"""
import pandas as pd
from .utilities import print_fun
from .mining import EnumeratedPattern

# Specific constants
dummy_col_name = 'dummy'
dummy_col_value = 'NA'


def mine_sequence_patterns(series_df: pd.DataFrame, nc_window_col: str,
                           support_threshold: float, crossk_threshold: float,
                           pattern_length: int, confidence_threshold: float = -1,
                           lag: int = 0, invalid_seq_indexes: list = []) -> pd.DataFrame:
    """Summary
    Main function / interface for the package
    """
    pass


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
    message = 'Formatting Enumerated Patterns as Output via: ({0})'.format(
        filter_type)
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
