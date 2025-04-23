from abc import ABC, abstractmethod
import time
from typing import Dict, Any, List, Tuple, Optional


class BaseModel(ABC):
    """Base class for all model implementations."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for the given prompt.

        Args:
            prompt: The input text prompt

        Returns:
            Tuple containing:
            - text_response: The full text response
            - metrics: Dictionary of performance metrics
            - audio_data: Audio data if applicable (or None)
        """
        pass

    def collect_metrics(self, start_time: float, first_token_time: Optional[float] = None) -> Dict[str, float]:
        """Collect basic timing metrics."""
        end_time = time.time()

        metrics = {
            "processing_time": end_time - start_time,
            "audio_duration": 0.0,  # Will be populated by specific model implementations
        }

        if first_token_time:
            metrics["time_to_audio_start"] = first_token_time - start_time

        return metrics
