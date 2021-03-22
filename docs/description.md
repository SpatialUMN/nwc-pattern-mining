## API interface for the package:
- **data**: DataFrame with only discretized candidate features and binarized anomalous window column
- **invalid_seq_indexes**: Indexes in data (starting from 0) across which sequence is invalid
- **nc_window_col_name**: Name of the column in data as the anomalous window
- **lag**: lag value to be considered for pattern mining
- **support_threshold**: Support threshold to be considered for pruning
- **crossk_threshold**: Cross-K threshold to be considered for pruning
- **patt_len**: Minimum pattern length to be mined