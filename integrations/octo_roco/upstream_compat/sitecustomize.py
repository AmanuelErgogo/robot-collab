"""Runtime shims for upstream Octo under newer dependency versions.

This module is imported automatically by Python when its directory is present
on ``PYTHONPATH``.  The launcher adds it only for the upstream Octo finetune
subprocess, leaving normal RoCo imports untouched.
"""

try:
    from jax.experimental.compilation_cache import compilation_cache

    if not hasattr(compilation_cache, "initialize_cache") and hasattr(
        compilation_cache, "set_cache_dir"
    ):
        compilation_cache.initialize_cache = compilation_cache.set_cache_dir
except Exception:
    pass

try:
    import jax
    from functools import wraps
    from jax.experimental import multihost_utils

    _process_allgather = multihost_utils.process_allgather

    @wraps(_process_allgather)
    def _process_allgather_tiled_default(in_tree, tiled=True):
        return _process_allgather(in_tree, tiled=tiled)

    multihost_utils.process_allgather = _process_allgather_tiled_default

    if not hasattr(jax.sharding, "PositionalSharding"):
        class _PositionalSharding:
            def __init__(self, *args, **kwargs):
                pass

            def replicate(self):
                return None

        jax.sharding.PositionalSharding = _PositionalSharding
except Exception:
    pass
