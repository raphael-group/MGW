"""Project-level JAX runtime defaults."""

import os

# Disable JAX GPU memory preallocation so OTT/JAX does not reserve most GPU
# memory in MGW workflows that also use PyTorch; set this env var before
# importing MGW to opt out and restore JAX's default allocator behavior.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
