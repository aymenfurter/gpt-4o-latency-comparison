import time
import requests
import json
import logging
from typing import Dict, Any, Tuple, Optional

from azure.identity.aio import DefaultAzureCredential

from models.base_model import BaseModel
from utils.exceptions import ModelGenerationError, ModelInitializationError
from utils.audio_utils import get_mp3_duration
from utils.client import create_azure_openai_client
from utils.metrics import calculate_tokens_per_second, create_error_metrics
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4O_DEPLOYMENT,
    TTS_DEPLOYMENT
)

# Configure logger
logger = logging.getLogger(__name__)


class GPT4OWhisperModel(BaseModel):
    """
    GPT-4o with Whisper TTS model implementation.

    Uses the GPT-4o deployment for text generation and Whisper for text-to-speech.
    """

    def __init__(self):
        """Initialize the GPT-4o with Whisper TTS model."""
        super().__init__("GPT-4o + Whisper TTS")
        self.gpt4o_deployment = GPT4O_DEPLOYMENT or "gpt-4o"
        self.tts_deployment = TTS_DEPLOYMENT or "tts"

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
                f"Failed to initialize GPT-4o + Whisper model: {e}") from e

    async def _get_tts_headers(self) -> Dict[str, str]:
        """
        Get the headers for the TTS API request.

        Returns:
            Dictionary of headers

        Raises:
            ModelInitializationError: If getting authentication fails
        """
        headers = {'Content-Type': 'application/json'}

        try:
            if AZURE_OPENAI_API_KEY:
                headers['api-key'] = AZURE_OPENAI_API_KEY
            else:
                # Use Microsoft Entra ID authentication for TTS
                credential = DefaultAzureCredential()
                token = await credential.get_token("https://cognitiveservices.azure.com/.default")
                headers['Authorization'] = f'Bearer {token.token}'

            return headers
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to get TTS authentication: {e}") from e

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response using GPT-4o and convert to speech using Whisper TTS.

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
            # Step 1: Generate text response with GPT-4o
            logger.debug(
                f"Generating text response with {self.gpt4o_deployment}")
            response = await self.client.chat.completions.create(
                model=self.gpt4o_deployment,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )

            text_response = response.choices[0].message.content
            text_complete_time = time.time()
            text_generation_time = text_complete_time - start_time

            logger.debug(
                f"Text response generated in {text_generation_time:.2f}s: {text_response[:50]}...")

            # Step 2: Convert text to speech using TTS
            tts_start_time = time.time()
            logger.debug(
                f"Converting text to speech with {self.tts_deployment}")

            # Get TTS headers with authentication
            headers = await self._get_tts_headers()

            # Prepare TTS API request
            tts_url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{self.tts_deployment}/audio/speech?api-version={AZURE_OPENAI_API_VERSION}"
            tts_body = {
                "input": text_response,
                "voice": "nova",
                "response_format": "mp3"
            }

            # Make TTS API request
            tts_response = requests.post(
                tts_url, headers=headers, data=json.dumps(tts_body), timeout=30
            )

            if tts_response.status_code != 200:
                raise ModelGenerationError(
                    f"TTS API error: {tts_response.status_code} - {tts_response.text}"
                )

            # Process audio data
            audio_data = tts_response.content
            tts_complete_time = time.time()
            tts_generation_time = tts_complete_time - tts_start_time

            logger.debug(f"Audio generated in {tts_generation_time:.2f}s")

            # Calculate audio duration
            audio_duration = get_mp3_duration(
                audio_data) if audio_data else 0.0

            # Collect metrics
            metrics = {
                "model": self.name,
                # Text generation metrics (the core GPT-4o part)
                "text_generation_time": text_generation_time,

                # TTS specific metrics
                "tts_time": tts_generation_time,

                # Combined metrics (total pipeline)
                "processing_time": text_generation_time + tts_generation_time,

                # Time until audio would start playing (text + TTS time)
                "time_to_audio_start": text_generation_time + tts_generation_time,

                # Actual duration of the generated audio
                "audio_duration": audio_duration,

                # Other metrics
                "token_count": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "audio_size_bytes": len(audio_data) if audio_data else 0
            }

            # Calculate tokens per second based on text generation time only
            metrics["tokens_per_second"] = calculate_tokens_per_second(
                metrics["token_count"], metrics["text_generation_time"]
            )

            logger.info(
                f"Generated GPT-4o + Whisper response in {metrics['processing_time']:.2f}s")

            return text_response, metrics, audio_data

        except Exception as e:
            error_msg = f"Error with GPT-4o + Whisper: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error metrics
            return error_msg, create_error_metrics(self.name, e, start_time), None
