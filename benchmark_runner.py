import time
import asyncio
import pandas as pd
import logging
from typing import List, Dict, Any, Tuple, Optional, Set

from models.base_model import BaseModel
from models.gpt4o_audio_model import GPT4OAudioModel
from models.gpt4o_realtime_model import GPT4ORealtimeModel
from models.gpt4o_whisper_model import GPT4OWhisperModel

from utils.exceptions import BenchmarkError
from config import DEFAULT_PROMPT, BENCHMARK_ITERATIONS

# Configure logger
logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmarks across multiple model implementations."""

    def __init__(self):
        """Initialize the benchmark runner with available models."""
        self.models: Dict[str, BaseModel] = {
            "gpt4o_realtime": GPT4ORealtimeModel(),
            "gpt4o_audio": GPT4OAudioModel(),
            "gpt4o_whisper": GPT4OWhisperModel()
        }
        self.results: Dict[str, List[Dict[str, Any]]] = {}

    async def initialize_models(self, model_keys: Optional[List[str]] = None) -> None:
        """
        Initialize the specified models or all models if none specified.

        Args:
            model_keys: List of model keys to initialize or None for all models
        """
        target_models = self._get_target_models(model_keys)

        for model_key, model in target_models.items():
            logger.info(f"Initializing model: {model.name}")
            try:
                await model.initialize()
            except Exception as e:
                logger.error(
                    f"Failed to initialize {model.name}: {e}", exc_info=True)

    async def run_benchmark(self,
                            prompt: Optional[str] = None,
                            iterations: Optional[int] = None,
                            selected_models: Optional[List[str]] = None) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]], Dict[str, bytes]]:
        """
        Run benchmarks for selected models with the given prompt.

        Args:
            prompt: Text prompt to use for benchmarking (uses DEFAULT_PROMPT if None)
            iterations: Number of iterations to run (uses BENCHMARK_ITERATIONS if None)
            selected_models: List of model keys to benchmark (or None for all)

        Returns:
            Tuple of (summary DataFrame, detailed metrics dict, audio samples dict)
        """
        # Set default values
        benchmark_prompt = prompt or DEFAULT_PROMPT
        benchmark_iterations = iterations or BENCHMARK_ITERATIONS

        # Get the models to benchmark
        target_models = self._get_target_models(selected_models)

        # Initialize containers for results
        all_metrics: Dict[str, List[Dict[str, Any]]] = {}
        all_audio: Dict[str, bytes] = {}

        # Initialize all selected models before starting benchmarks
        await self.initialize_models(selected_models)

        # Run benchmarks for each model
        for model_key, model in target_models.items():
            logger.info(
                f"Benchmarking {model.name} ({iterations} iterations)...")
            model_metrics = []

            for i in range(benchmark_iterations):
                logger.info(f"  Iteration {i+1}/{benchmark_iterations}")
                try:
                    text, metrics, audio = await model.generate_response(benchmark_prompt)

                    # Add metadata to metrics
                    metrics["model"] = model.name
                    metrics["iteration"] = i+1
                    metrics["text_length"] = len(text)

                    model_metrics.append(metrics)

                    # Save only the last audio sample for each model
                    if audio:
                        all_audio[model_key] = audio

                    logger.debug(f"  Completed with metrics: {metrics}")

                except Exception as e:
                    logger.error(
                        f"Error benchmarking {model.name}: {e}", exc_info=True)

            # Store metrics for this model
            all_metrics[model_key] = model_metrics

        # Create summary dataframe with aggregated results
        summary_df = self._create_summary_dataframe(all_metrics)

        return summary_df, all_metrics, all_audio

    def _get_target_models(self, selected_models: Optional[List[str]] = None) -> Dict[str, BaseModel]:
        """
        Get the target models to benchmark based on selection.

        Args:
            selected_models: List of model keys to include or None for all

        Returns:
            Dictionary of selected model instances
        """
        if selected_models is None:
            return self.models

        # Filter to only the selected models
        return {k: v for k, v in self.models.items() if k in selected_models}

    def _create_summary_dataframe(self, metrics_dict: Dict[str, List[Dict[str, Any]]]) -> pd.DataFrame:
        """
        Create a summary dataframe from the collected metrics.

        Args:
            metrics_dict: Dictionary of metrics by model

        Returns:
            Pandas DataFrame with average metrics for each model
        """
        summary_data = []

        # Calculate average metrics for each model
        for model_key, metrics_list in metrics_dict.items():
            if not metrics_list:
                continue

            model_name = self.models[model_key].name
            metrics_count = len(metrics_list)

            # Calculate average values for key metrics
            avg_metrics = {
                "model": model_name,
                "processing_time": sum(m.get("processing_time", 0) for m in metrics_list) / metrics_count,
                "time_to_audio_start": sum(m.get("time_to_audio_start", 0) for m in metrics_list) / metrics_count,
                "audio_duration": sum(m.get("audio_duration", 0) for m in metrics_list) / metrics_count,
                "tokens_per_second": sum(m.get("tokens_per_second", 0) for m in metrics_list) / metrics_count,
            }

            # Add audio-specific metrics if available
            if "audio_size_bytes" in metrics_list[0]:
                avg_metrics["audio_size_bytes"] = sum(
                    m["audio_size_bytes"] for m in metrics_list) / metrics_count

            # Add model-specific metrics if present in all instances
            model_specific_keys = self._get_common_metric_keys(metrics_list)
            for key in model_specific_keys:
                if key not in avg_metrics:
                    avg_metrics[key] = sum(m.get(key, 0)
                                           for m in metrics_list) / metrics_count

            summary_data.append(avg_metrics)

        return pd.DataFrame(summary_data)

    def _get_common_metric_keys(self, metrics_list: List[Dict[str, Any]]) -> Set[str]:
        """
        Get the set of metric keys that appear in all metrics dictionaries.

        Args:
            metrics_list: List of metrics dictionaries

        Returns:
            Set of common keys
        """
        if not metrics_list:
            return set()

        # Start with all keys from the first metrics dict
        common_keys = set(metrics_list[0].keys())

        # Intersect with keys from each subsequent dict
        for metrics in metrics_list[1:]:
            common_keys &= set(metrics.keys())

        return common_keys
