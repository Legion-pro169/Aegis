"""
Shared Utilities Module
=======================

Common utilities used across Aegis modules:
- Logging configuration
- Configuration management
- Data validation
- Performance timing
"""

import logging
import time
from functools import wraps


def setup_logging(level=logging.INFO):
    """Configure logging for Aegis."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('aegis')


def timer(func):
    """Decorator to time function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} completed in {elapsed:.2f}s")
        return result
    return wrapper


__all__ = ['setup_logging', 'timer']
