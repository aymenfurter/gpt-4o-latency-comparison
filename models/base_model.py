from abc import ABC, abstractmethod
import time
import logging
from typing import Dict, Any, Tuple, Optional

from utils.exceptions import ModelGenerationError

# Configure logger
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base class for all model implementations."""

    def __init__(self, name: str):
        """
        Initialize the model.

        Args:
            name: Human-readable name of the model
        """
        self.name = name
        self.client = None
        self.initialized = False

    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the model's client and resources.

        This method should be called before generate_response.

        Raises:
            ModelInitializationError: If initialization fails
        """
        pass

    @abstractmethod
    async def generate_response_from_audio(self, audio_data: bytes, text_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for the given audio input.

        Args:
            audio_data: The input audio data (WAV format)
            text_prompt: Optional text prompt to accompany the audio

        Returns:
            Tuple containing:
            - text_response: The full text response
            - metrics: Dictionary of performance metrics
            - audio_data: Audio data if applicable (or None)

        Raises:
            ModelGenerationError: If response generation fails
        """
        pass

    async def ensure_initialized(self) -> None:
        """
        Ensure the model is initialized before use.

        Calls initialize() if the model is not already initialized.

        Raises:
            ModelInitializationError: If initialization fails
        """
        if not self.initialized:
            await self.initialize()
            self.initialized = True

    def collect_basic_metrics(self, start_time: float, first_token_time: Optional[float] = None) -> Dict[str, Any]:
        """
        Collect basic timing metrics.

        Args:
            start_time: Time when processing started
            first_token_time: Time when first token was received (optional)

        Returns:
            Dictionary containing basic timing metrics
        """
        end_time = time.time()

        metrics = {
            "model": self.name,
            "processing_time": end_time - start_time,
            "audio_duration": 0.0,  # Will be populated by specific model implementations
        }

        if first_token_time:
            metrics["time_to_audio_start"] = first_token_time - start_time
        else:
            metrics["time_to_audio_start"] = 0.0

        return metrics
