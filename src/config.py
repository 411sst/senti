"""Project-wide configuration constants.

Import ``SEED`` everywhere randomness is involved (train/test splits, model
initialisation, sampling) so every run is reproducible from a single source of
truth.
"""

SEED = 42
