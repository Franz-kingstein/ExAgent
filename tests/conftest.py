"""Shared pytest configuration.

USE_TORCH=0 must be set before any langchain/transformers import: the shared
site-packages on this machine contains a CUDA-only torch build that fails to
import (missing libcublasLt). Setting the flag makes transformers skip torch.
"""

import os

os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
