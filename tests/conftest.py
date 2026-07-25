"""Pytest session setup.

**PyTorch and XGBoost fight over OpenMP on macOS.** Each bundles its own OpenMP
runtime, and with both loaded in one process the second one to run either
segfaults or deadlocks -- we saw XGBoost segfault in ``_meta_from_numpy`` when
torch was imported first, and torch segfault in ``Conv1d`` when xgboost was
imported first, so import order alone does not fix it. Empirically:

    OMP_NUM_THREADS unset          -> deadlock
    KMP_DUPLICATE_LIB_OK=TRUE      -> deadlock
    OMP_NUM_THREADS=1              -> both work

Pinning each runtime to a single thread stops them contending. This must be set
BEFORE either library is imported, which is why it lives in conftest (imported
before any test module) rather than in a test file.

Only the test suite is constrained this way -- training scripts import one
library at a time and keep full threading. A script that genuinely needs both
should set the same variable at the top, before its imports.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch  # noqa: E402,F401
import xgboost  # noqa: E402,F401
