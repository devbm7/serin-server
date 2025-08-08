#!/usr/bin/env python3
"""
Timing utilities for performance analysis and monitoring.
"""

import time
import logging
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class PerformanceTimer:
    """A timer class for tracking performance metrics."""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.duration = None
        
    def start(self):
        """Start the timer."""
        self.start_time = time.time()
        logger.info(f"[{self.name}] Timer started")
        
    def stop(self):
        """Stop the timer and calculate duration."""
        if self.start_time is None:
            logger.warning(f"[{self.name}] Timer was not started")
            return None
            
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        logger.info(f"[{self.name}] Timer stopped: {self.duration:.3f}s")
        return self.duration
        
    def get_duration(self) -> Optional[float]:
        """Get the duration if timer has been stopped."""
        return self.duration

@contextmanager
def timed_operation(name: str, log_level: str = "info"):
    """Context manager for timing operations."""
    start_time = time.time()
    logger.info(f"Starting {name}")
    
    try:
        yield
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"{name} failed after {duration:.3f}s: {e}")
        raise
    else:
        duration = time.time() - start_time
        if log_level == "info":
            logger.info(f"{name} completed in {duration:.3f}s")
        elif log_level == "debug":
            logger.debug(f"{name} completed in {duration:.3f}s")

class PerformanceMonitor:
    """Monitor for tracking multiple performance metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.current_timers: Dict[str, float] = {}
        
    def start_timer(self, name: str):
        """Start a timer for a specific operation."""
        self.current_timers[name] = time.time()
        
    def stop_timer(self, name: str) -> float:
        """Stop a timer and record the duration."""
        if name not in self.current_timers:
            logger.warning(f"Timer '{name}' was not started")
            return 0.0
            
        duration = time.time() - self.current_timers[name]
        del self.current_timers[name]
        
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        logger.info(f"{name} completed in {duration:.3f}s")
        return duration
        
    def get_average(self, name: str) -> float:
        """Get average duration for a specific operation."""
        if name not in self.metrics or not self.metrics[name]:
            return 0.0
        return sum(self.metrics[name]) / len(self.metrics[name])
        
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get a summary of all recorded metrics."""
        summary = {}
        for name, durations in self.metrics.items():
            if durations:
                summary[name] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'average': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        return summary
        
    def print_summary(self):
        """Print a formatted summary of all metrics."""
        summary = self.get_summary()
        if not summary:
            logger.info("No performance metrics recorded")
            return
            
        logger.info("=== PERFORMANCE SUMMARY ===")
        for name, stats in summary.items():
            logger.info(f"{name}:")
            logger.info(f"  Count: {stats['count']}")
            logger.info(f"  Total: {stats['total']:.3f}s")
            logger.info(f"  Average: {stats['average']:.3f}s")
            logger.info(f"  Min: {stats['min']:.3f}s")
            logger.info(f"  Max: {stats['max']:.3f}s")
        logger.info("============================")

def format_duration(seconds: float) -> str:
    """Format duration in a human-readable way."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"

def log_performance_breakdown(operation_name: str, breakdown: Dict[str, float]):
    """Log a performance breakdown for an operation."""
    total_time = sum(breakdown.values())
    logger.info(f"=== {operation_name.upper()} PERFORMANCE BREAKDOWN ===")
    for component, duration in breakdown.items():
        percentage = (duration / total_time) * 100 if total_time > 0 else 0
        logger.info(f"{component}: {duration:.3f}s ({percentage:.1f}%)")
    logger.info(f"Total: {total_time:.3f}s")
    logger.info("=" * (len(operation_name) + 30)) 