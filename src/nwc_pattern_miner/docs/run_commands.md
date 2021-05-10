# Testing various modules:
- **stategraph**: `python -m tests.patterncount.test_stategraph`
- **latticegraph**: `python -m tests.mining.test_latticegraph`
- **candidatepattern**: `python -m tests.mining.test_candidatepattern`
- **patternclient**: `python -m tests.test_pattern_mining_client`
- **supportpruning**: `python -m tests.pruning.test_support_pruning`
- **ubpruning**: `python -m tests.pruning.test_ub_pruning`
- **patternminer**: `python -m tests.test_patternminer`

# Building package:
- `python -m build`: to load and build changes
- `python -m pip install dist/nwc_pattern_miner...whl`: to install local copy
- `python -m twine upload dist/*`: to upload build and ready package onto pypi