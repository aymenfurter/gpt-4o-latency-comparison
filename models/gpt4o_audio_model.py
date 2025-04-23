import time
import base64
import logging
from typing import Dict, Any, Tuple, Optional

from models.base_model import BaseModel
from utils.exceptions import ModelGenerationError, ModelInitializationError
from utils.audio_utils import get_wav_duration
from utils.client import create_azure_openai_client
from utils.metrics import calculate_tokens_per_second, create_error_metrics
from config import (
    AZURE_OPENAI_API_VERSION,
    GPT4O_AUDIO_DEPLOYMENT
)

# Configure logger
logger = logging.getLogger(__name__)


class GPT4OAudioModel(BaseModel):
    """
    GPT-4o Audio model implementation.

    Uses the GPT-4o Audio Preview deployment to generate text and audio simultaneously.
    """

    def __init__(self):
        """Initialize the GPT-4o Audio model."""
        super().__init__("GPT-4o Audio Preview")
        self.deployment_name = GPT4O_AUDIO_DEPLOYMENT or "gpt-4o-audio-preview"

    async def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client.

        Raises:
            ModelInitializationError: If client initialization fails
        """
        try:
            self.client = await create_azure_openai_client(AZURE_OPENAI_API_VERSION)
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize GPT-4o Audio model: {e}") from e

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for the given prompt with both text and audio.

        Args:
            prompt: The input text prompt

        Returns:
            Tuple containing text response, metrics, and audio data

        Raises:
            ModelGenerationError: If response generation fails
        """
        await self.ensure_initialized()
        start_time = time.time()

        try:
            # Make the audio chat completions request
            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                modalities=["text", "audio"],
                audio={"voice": "alloy", "format": "wav"},
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            # Audio model doesn't stream, so first token is entire response
            completion_time = time.time()
            processing_time = completion_time - start_time

            text_response = response.choices[0].message.audio.transcript
            audio_data = base64.b64decode(
                response.choices[0].message.audio.data)

            # Calculate actual audio duration from the WAV file
            audio_duration = get_wav_duration(
                audio_data) if audio_data else 0.0
            token_count = response.usage.completion_tokens

            # Collect metrics
            metrics = {
                "model": self.name,
                "text_generation_time": processing_time,
                "time_to_audio_start": processing_time,
                "processing_time": processing_time,
                "audio_duration": audio_duration,
                "token_count": token_count,
                "total_tokens": response.usage.total_tokens,
                "audio_size_bytes": len(audio_data)
            }

            # Calculate tokens per second
            metrics["tokens_per_second"] = calculate_tokens_per_second(
                token_count, processing_time
            )

            logger.info(
                f"Generated response with GPT-4o Audio in {processing_time:.2f}s")
            return text_response, metrics, audio_data

        except Exception as e:
            error_msg = f"Error with GPT-4o Audio: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error metrics
            return error_msg, create_error_metrics(self.name, e, start_time), None
