# SPDX-License-Identifier: Apache-2.0
"""Release the CPU client after compilation without replacing live TPU clients.

JAX has no selective backend-release API. Its private registry needs a lookup
shim because ordinary initialization does not retry CPU while TPU is registered.
Remove this shim when JAX exposes selective release and lazy reinitialization.
"""

from jax._src import xla_bridge


class _CpuBackendLookup:
    def __init__(self, lookup):
        self.lookup = lookup

    def __call__(self, platform=None):
        if platform is None or isinstance(platform, str):
            requested = (
                platform
                or xla_bridge._XLA_BACKEND.value
                or xla_bridge._PLATFORM_NAME.value
            )
            if requested == "cpu":
                with xla_bridge._backend_lock:
                    if "cpu" not in xla_bridge._backends:
                        xla_bridge._backends["cpu"] = xla_bridge._init_backend("cpu")
        return self.lookup(platform)


def release_cpu_backend() -> bool:
    """Drop registry references at the idle, post-warmup cleanup boundary.

    Call after clearing JAX caches and before collecting/freezing GC. Existing
    arrays or executables keep their clients alive. Preserve CPU when it is the
    default backend, and never initialize a backend just to release it.
    """
    with xla_bridge._backend_lock:
        backend = xla_bridge._backends.get("cpu")
        if backend is None or backend is xla_bridge._default_backend:
            return False
        if not isinstance(xla_bridge._get_backend_uncached, _CpuBackendLookup):
            xla_bridge._get_backend_uncached = _CpuBackendLookup(
                xla_bridge._get_backend_uncached
            )
        del xla_bridge._backends["cpu"]
        xla_bridge.get_backend.cache_clear()
    return True
