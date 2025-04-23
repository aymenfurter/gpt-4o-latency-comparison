import time
import asyncio
import pandas as pd
import traceback
from typing import List, Dict, Any, Tuple, Optional

from models.base_model import BaseModel
from models.gpt4o_audio_model import GPT4OAudioModel
from models.gpt4o_realtime_model import GPT4ORealtimeModel
from models.gpt4o_whisper_model import GPT4OWhisperModel

from config import DEFAULT_PROMPT, BENCHMARK_ITERATIONS


class BenchmarkRunner:
    """Runs benchmarks across multiple model implementations."""

    def __init__(self):
        self.models = {
            "gpt4o_realtime": GPT4ORealtimeModel(),
            "gpt4o_audio": GPT4OAudioModel(),
            "gpt4o_whisper": GPT4OWhisperModel()
        }
        self.results = {}

    async def run_benchmark(self,
                            prompt: str = None,
                            iterations: int = None,
                            selected_models: List[str] = None) -> Tuple[pd.DataFrame, Dict[str, List[Dict[str, Any]]], Dict[str, bytes]]:
        """
        Run benchmarks for all models with the given prompt.

        Args:
            prompt: Text prompt to use for benchmarking
            iterations: Number of iterations to run
            selected_models: List of model keys to benchmark (or None for all)

        Returns:
            Tuple of (summary DataFrame, detailed metrics dict, audio samples dict)
        """
        if prompt is None:
            prompt = DEFAULT_PROMPT

        if iterations is None:
            iterations = BENCHMARK_ITERATIONS

        active_models = {k: v for k, v in self.models.items()
                         if selected_models is None or k in selected_models}

        all_metrics = {}
        all_audio = {}

        for model_key, model in active_models.items():
            print(f"Benchmarking {model.name}...")
            model_metrics = []

            for i in range(iterations):
                print(f"  Iteration {i+1}/{iterations}")
                try:
                    text, metrics, audio = await model.generate_response(prompt)
                    model_metrics.append(metrics)

                    # Save only the last audio sample
                    if audio:
                        all_audio[model_key] = audio

                    # Add metadata to metrics
                    metrics["model"] = model.name
                    metrics["iteration"] = i+1
                    metrics["text_length"] = len(text)

                except Exception as e:
                    print(f"Error benchmarking {model.name}: {str(e)}")
                    traceback.print_exc()  # Print the full traceback

            all_metrics[model_key] = model_metrics

        # Create summary dataframe
        summary_data = []

        for model_key, metrics_list in all_metrics.items():
            if not metrics_list:
                continue

            model_name = self.models[model_key].name
            avg_metrics = {
                "model": model_name,
                "processing_time": sum(m.get("processing_time", 0) for m in metrics_list) / len(metrics_list),
                "time_to_audio_start": sum(m.get("time_to_audio_start", 0) for m in metrics_list) / len(metrics_list),
                "audio_duration": sum(m.get("audio_duration", 0) for m in metrics_list) / len(metrics_list),
                "tokens_per_second": sum(m.get("tokens_per_second", 0) for m in metrics_list) / len(metrics_list),
            }

            # Add audio-specific metrics if available
            if "audio_size_bytes" in metrics_list[0]:
                avg_metrics["audio_size_bytes"] = sum(
                    m["audio_size_bytes"] for m in metrics_list) / len(metrics_list)

            summary_data.append(avg_metrics)

        summary_df = pd.DataFrame(summary_data)
        return summary_df, all_metrics, all_audio
