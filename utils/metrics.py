"""Utility functions for collecting and processing metrics."""

import time
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List


def calculate_tokens_per_second(token_count: int, processing_time: float) -> float:
    """
    Calculate tokens per second rate.

    Args:
        token_count: Number of tokens generated
        processing_time: Time taken to generate tokens in seconds

    Returns:
        Tokens per second rate or 0 if inputs are invalid
    """
    if token_count <= 0 or processing_time <= 0:
        return 0.0
    return token_count / processing_time


def create_error_metrics(model_name: str, error: Exception, start_time: float) -> Dict[str, Any]:
    """
    Create standardized error metrics dictionary.

    Args:
        model_name: Name of the model that encountered the error
        error: The exception that occurred
        start_time: Time when processing started

    Returns:
        Dictionary with error metrics
    """
    return {
        "model": model_name,
        "error": str(error),
        "processing_time": time.time() - start_time,
        "time_to_audio_start": 0,
        "audio_duration": 0,
        "token_count": 0,
        "total_tokens": 0,
        "tokens_per_second": 0,
        "audio_size_bytes": 0
    }


def calculate_relative_performance(df: pd.DataFrame, column: str, higher_is_better: bool = False) -> pd.Series:
    """
    Calculate relative performance percentages for a metric column.

    Args:
        df: DataFrame containing the metrics
        column: Name of the metric column
        higher_is_better: Whether higher values indicate better performance

    Returns:
        Series with relative performance percentages
    """
    if column not in df.columns or df.empty:
        return pd.Series([])

    # Get valid values (non-NaN)
    valid_values = df[column].dropna()

    if valid_values.empty:
        return pd.Series([])

    if higher_is_better:
        # For metrics where higher is better (e.g., tokens_per_second)
        best_value = valid_values.max()
        relative_perf = (df[column] / best_value) * 100
    else:
        # For metrics where lower is better (e.g., latency)
        best_value = valid_values.min()
        relative_perf = (best_value / df[column]) * 100

    return relative_perf


def format_metric_value(value: Any, metric_type: str = "time") -> str:
    """
    Format a metric value for display.

    Args:
        value: The metric value to format
        metric_type: Type of metric ('time', 'tokens', 'bytes', etc.)

    Returns:
        Formatted string representation
    """
    if pd.isna(value) or value is None:
        return "Not applicable"

    if not isinstance(value, (int, float)):
        return str(value)

    if metric_type == "time":
        return f"{value:.3f}s"
    elif metric_type == "tokens_per_second":
        return f"{value:.1f}"
    elif metric_type == "bytes":
        # Convert to KB for better readability
        kb_value = value / 1024
        return f"{kb_value:.1f} KB"
    else:
        # Default formatting for other numeric values
        if isinstance(value, int):
            return str(value)
        else:
            return f"{value:.2f}"
