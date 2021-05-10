# nwc-pattern-mining
Non-compliant Window Co-occurrence pattern mining in temporal data

- A Python library to find sequential association patterns in time-series data, co-occurring with target anomalous windows.
- Anomalous windows are sequences in which a target variable defies expected behavior (e.g. Emissions from a vehicle).
- The patterns are found in co-occurrence with a specific non-compliant feature (to decipher the reason behinds it's irregular behavior).
- The package uses various pruning methodologies to speed up pattern mining, and hashing for quick support count.
- The algorithm is based on research: (Discovering non-compliant window co-occurrence patterns)[https://link.springer.com/article/10.1007/s10707-016-0289-3]

# API-description:
`from nwc_pattern_miner import mine_sequence_patterns`

## API Parameters: 
- **series_df**: `pd.DataFrame`; Input DataFrame (Only features [discretized] and Target [binarized] columns)  
- **nc_window_col**: `str`; Column Name with Binary Target (Anomalous Windows)
- **support_threshold**: `float`; Support threshold for sequence co-occurrence patterns
- **crossk_threshold**: `float`; Ripley's Cross-k threshold for sequence co-occurrence patterns
- **pattern_length**: `int`; length of feature sequences co-occurring with anomalous windows
- **confidence_threshold**: `float, default=-1`; Confidence threshold for sequence co-occurrence patterns
- **lag**: `int, default= 0`; lag consideration between sequence patterns and anomalous windows
- **invalid_seq_indexes**: `list, default=list()`; list of indexes across which sequence patterns would be invalidated
- **output_metric**: `{'crossk', 'support'}, default='crossk'`; Metric used to sort patterns mined
- **output_type**: `{'topk', 'threshold'}, default='topk';` Type of output for sequence patterns mined
- **output_threshold**: `float, default= -1`; Threshold cutoff used to get output sequence patterns, if `output_type='threshold'`
- **topk**: `int, default=100`; Top-k sequence patterns obtained based on `output_metric`, if `output_type='topk'`
- **pruning_type**: `str, default=apriori`; Between `[apriori, br-dr]`, both have same run-time, 'br-dr' does more enumerations but enumeration speed is much faster due to UB pruning. 

## Sample Input DataFrame:
| engrpm      | EGRkgph     | MSPhum      | EngTq       | NCWindow    |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| 9           | 11          | 5           | 3           | 1           |
| 3           | 1           | 5           | 4           | 0           |


## Sample Output DataFrame:
| engrpm | EGRkgph | MSPhum | EngTq | Count | Support | Kvalue | Confidence | First Occurrence Index |
| -------| --------| -------| ------| ------| --------|--------|------------|------------------------|
|        | 4 4 4   | 5 5 5  | 2 2 2 | 146   | 0.00528 | 2.377  | 1.0        | 47167                  |
|        | 4 4 4   |        | 7 7 7 | 250   | 0.00643 | 2.357  | 1.0        | 41984                  |