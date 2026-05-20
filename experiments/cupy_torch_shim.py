"""
cupy_torch_shim.py — Drop-in CuPy replacement using PyTorch tensors.

Implements the subset of CuPy's API used by exp_extended_epsilon.py so the
experiment can run on any pod that has PyTorch + CUDA without requiring CuPy.

Covered API surface:
  cp.float32(x)                    → Python float (scalar)
  cp.random.seed(s)                → torch.manual_seed
  cp.random.standard_normal(shape) → torch.randn on CUDA
  cp.zeros(n, dtype=...)           → torch.zeros on CUDA
  cp.maximum(a, b)                 → torch.clamp_min / torch.maximum
  cp.asarray(x, dtype=...)         → np.asarray of cpu tensor
  cp.cuda.runtime.getDeviceProperties(id) → torch device props dict
  tensor.get()                     → tensor.cpu().numpy()  (monkey-patched)
  __version__                      → "torch-shim/<torch_version>"
"""

import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

__version__: str = f"torch-shim/{torch.__version__}"

# Maximum bytes to allocate for a single noise tensor (4 GiB).
# Larger shapes use the lazy row-by-row generator to avoid OOM.
_MAX_ALLOC_BYTES: int = 4 * 1024 ** 3


# ---------------------------------------------------------------------------
# Dtype aliases — identical to numpy equivalents so they work as both
# callables (cp.float32(scalar)) and dtype arguments (dtype=cp.float64).
# ---------------------------------------------------------------------------
float32 = np.float32
float64 = np.float64


# ---------------------------------------------------------------------------
# Lazy noise matrix — generates rows on demand to avoid OOM on tight ε
# ---------------------------------------------------------------------------
class _LazyNoiseMatrix:
    """Stands in for a pre-allocated (n_timesteps, n_paths) float32 tensor.

    Instead of allocating everything at once, each row is generated when
    indexed.  Because PyTorch's RNG is stateful, iterating rows 0…T-1 in
    order produces the *same* random sequence as a single large allocation
    would, so results are reproducible when the seed is set beforehand.
    """

    def __init__(self, n_timesteps: int, n_paths: int, scale: float = 1.0) -> None:
        self.n_timesteps = n_timesteps
        self.n_paths = n_paths
        self.scale = scale

    def __mul__(self, scalar: float) -> "_LazyNoiseMatrix":
        return _LazyNoiseMatrix(self.n_timesteps, self.n_paths, float(scalar))

    def __rmul__(self, scalar: float) -> "_LazyNoiseMatrix":
        return self.__mul__(scalar)

    def __getitem__(self, key) -> torch.Tensor:
        if isinstance(key, slice):
            start, stop, _ = key.indices(self.n_timesteps)
            rows = torch.randn(stop - start, self.n_paths, device=DEVICE, dtype=torch.float32)
            return rows * self.scale if self.scale != 1.0 else rows
        # integer index — generate one row
        row = torch.randn(self.n_paths, device=DEVICE, dtype=torch.float32)
        return row * self.scale if self.scale != 1.0 else row


# ---------------------------------------------------------------------------
# Random module
# ---------------------------------------------------------------------------
class random:  # noqa: N801
    @staticmethod
    def seed(s: int) -> None:
        torch.manual_seed(s)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(s)

    @staticmethod
    def standard_normal(shape, dtype=None):
        """Return noise tensor, or a lazy row-generator if it would OOM."""
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            n_timesteps, n_paths = shape
            mem_bytes = n_timesteps * n_paths * 4  # float32
            if mem_bytes > _MAX_ALLOC_BYTES:
                return _LazyNoiseMatrix(n_timesteps, n_paths)
        return torch.randn(shape, device=DEVICE, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Array constructors
# ---------------------------------------------------------------------------
def zeros(n, dtype=None) -> torch.Tensor:
    shape = (n,) if isinstance(n, int) else tuple(int(d) for d in n)
    return torch.zeros(shape, device=DEVICE, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Elementwise ops
# ---------------------------------------------------------------------------
def maximum(a: torch.Tensor, b) -> torch.Tensor:
    if isinstance(b, (int, float)):
        return torch.clamp_min(a, b)
    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b, device=DEVICE, dtype=a.dtype)
    return torch.maximum(a, b)


def sum(a: torch.Tensor, axis=None) -> torch.Tensor:  # noqa: A001
    if axis is None:
        return torch.sum(a)
    return torch.sum(a, dim=axis)


_NP_TO_TORCH = {
    np.float32: torch.float32,
    np.float64: torch.float64,
    np.int32: torch.int32,
    np.int64: torch.int64,
}


def asarray(x, dtype=None) -> torch.Tensor:
    """Return x as a torch tensor on DEVICE (with optional dtype cast).

    Keeps the result as a torch.Tensor so that .get() (monkey-patched above)
    works uniformly whether the caller came via cp.asarray() or xp.asarray().
    """
    if isinstance(x, torch.Tensor):
        t = x
    else:
        arr = np.asarray(x)
        t = torch.from_numpy(arr).to(DEVICE)

    if dtype is not None:
        torch_dtype = _NP_TO_TORCH.get(dtype)
        if torch_dtype is not None:
            t = t.to(dtype=torch_dtype)
    return t


# ---------------------------------------------------------------------------
# CUDA device info
# ---------------------------------------------------------------------------
class cuda:  # noqa: N801
    class runtime:  # noqa: N801
        @staticmethod
        def getDeviceProperties(device_id: int) -> dict:
            if not torch.cuda.is_available():
                return {"name": b"CPU (no CUDA)"}
            props = torch.cuda.get_device_properties(device_id)
            return {"name": props.name.encode()}


# ---------------------------------------------------------------------------
# Monkey-patch torch.Tensor with .get() → numpy (matches CuPy's API)
# ---------------------------------------------------------------------------
if not hasattr(torch.Tensor, "get"):
    torch.Tensor.get = lambda self: self.cpu().numpy()  # type: ignore[assignment]

# Note: np.ndarray is an immutable C type — cannot monkey-patch .get() on it.
# asarray() always returns a torch.Tensor, so .get() is always available.
