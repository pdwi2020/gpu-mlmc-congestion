"""
Utility functions for GPU-Accelerated MLMC Network Modeling.

Provides common helpers used across modules:
- Seed management for reproducibility
- Timing utilities for benchmarking
- Logging configuration
- Path management
- Numerical utilities
"""

import numpy as np
import time
import logging
from typing import Optional, Generator, Any, Dict
from contextlib import contextmanager
from pathlib import Path
from functools import wraps

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """
    Set seeds for all random number generators for reproducibility.

    Args:
        seed: Integer seed value
    """
    np.random.seed(seed)

    # Try to set PyTorch seed if available
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    # Try to set CuPy seed if available
    try:
        import cupy
        cupy.random.seed(seed)
    except ImportError:
        pass

    logger.debug(f"Set all random seeds to {seed}")


@contextmanager
def timer(name: str = "Operation") -> Generator[None, None, None]:
    """
    Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed

    Yields:
        None

    Example:
        with timer("Matrix multiplication"):
            result = A @ B
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{name}: {elapsed:.4f}s")


def ensure_directory(path: str) -> Path:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path (string or Path)

    Returns:
        Path object for the directory
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string (optional)
        log_file: Path to log file (optional, logs to console if None)
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    handlers = [logging.StreamHandler()]

    if log_file is not None:
        ensure_directory(str(Path(log_file).parent))
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    default: float = 0.0
) -> np.ndarray:
    """
    Perform element-wise division with handling for division by zero.

    Args:
        numerator: Numerator array
        denominator: Denominator array
        default: Value to use when denominator is zero

    Returns:
        Result array with safe division
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(
            denominator != 0,
            numerator / denominator,
            default
        )
    return result


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Compute moving average of a 1D array.

    Args:
        data: Input array
        window: Window size for averaging

    Returns:
        Moving average array (length = len(data) - window + 1)
    """
    if window < 1:
        raise ValueError("Window size must be at least 1")
    if window > len(data):
        raise ValueError("Window size cannot exceed data length")

    return np.convolve(data, np.ones(window) / window, mode='valid')


def compute_statistics(samples: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive statistics for a sample array.

    Args:
        samples: 1D array of samples

    Returns:
        Dictionary with mean, std, var, min, max, median, and quartiles
    """
    return {
        'mean': float(np.mean(samples)),
        'std': float(np.std(samples, ddof=1)),
        'var': float(np.var(samples, ddof=1)),
        'min': float(np.min(samples)),
        'max': float(np.max(samples)),
        'median': float(np.median(samples)),
        'q25': float(np.percentile(samples, 25)),
        'q75': float(np.percentile(samples, 75)),
        'n_samples': len(samples)
    }


def format_scientific(value: float, precision: int = 2) -> str:
    """
    Format a number in scientific notation.

    Args:
        value: Number to format
        precision: Number of decimal places

    Returns:
        Formatted string
    """
    return f"{value:.{precision}e}"


def format_time(seconds: float) -> str:
    """
    Format time duration in human-readable form.

    Args:
        seconds: Duration in seconds

    Returns:
        Formatted string (e.g., "2.5s", "1m 30s", "2h 15m")
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def retry_on_error(max_retries: int = 3, delay: float = 1.0):
    """
    Decorator to retry a function on error.

    Args:
        max_retries: Maximum number of retry attempts
        delay: Delay between retries in seconds

    Returns:
        Decorated function
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt + 1}/{max_retries + 1}): {e}"
                        )
                        time.sleep(delay)
                    else:
                        logger.error(f"{func.__name__} failed after {max_retries + 1} attempts")
            raise last_error
        return wrapper
    return decorator


def check_gpu_available() -> bool:
    """
    Check if GPU (CUDA) is available.

    Returns:
        True if CUDA is available, False otherwise
    """
    try:
        import pycuda.driver as cuda
        cuda.init()
        return cuda.Device.count() > 0
    except Exception:
        pass

    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        pass

    try:
        import cupy
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        pass

    return False


def get_memory_usage() -> Dict[str, float]:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory usage in MB
    """
    import os

    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        return {
            'rss_mb': mem_info.rss / (1024 * 1024),
            'vms_mb': mem_info.vms / (1024 * 1024)
        }
    except ImportError:
        return {'rss_mb': -1, 'vms_mb': -1}


# Export public API
__all__ = [
    'set_all_seeds',
    'timer',
    'ensure_directory',
    'setup_logging',
    'safe_divide',
    'moving_average',
    'compute_statistics',
    'format_scientific',
    'format_time',
    'retry_on_error',
    'check_gpu_available',
    'get_memory_usage',
]
