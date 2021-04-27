# Contains common utility functions
import pandas as pd


def print_fun(message: str, status: str = '') -> None:
    """Summary
        For printing updates across package
    """
    separator = '-'
    custom_msg = '\n' + '{1}' * 20 + ' | {0}' + ' | ' + '{1}' * 20 + '\n'
    if status == 'step':
        separator = '*'

    print(custom_msg.format(message, separator))


def stringify_dataframe(data: pd.DataFrame, start_idx: int = -1, end_idx: int = -1) -> str:
    """Summary
        Used to stringify dataframes for hashing
    """
    # If need to stringify complete dataframe
    if start_idx != -1:
        data = data.iloc[start_idx: end_idx]

    return data.to_json(orient='records')
