import time
import requests
import json
import logging
import io
import base64
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
    MINITTS_OPENAI_ENDPOINT,
    MINITTS_OPENAI_API_KEY,
    MINITTS_OPENAI_API_VERSION,
    GPT4O_DEPLOYMENT,
    GPT4O_MINI_TTS_DEPLOYMENT,
    GPT4O_MINI_TRANSCRIBE_DEPLOYMENT
)

# Configure logger
logger = logging.getLogger(__name__)


class GPT4OMiniTTSMiniTranscribeModel(BaseModel):
    """
    GPT-4o with GPT-4o-mini-transcribe and GPT-4o-mini-tts model implementation.

    Uses the GPT-4o-mini-transcribe deployment for transcription, 
    GPT-4o for text generation, and GPT-4o-mini-tts for text-to-speech.
    """

    def __init__(self):
        """Initialize the GPT-4o with GPT-4o-mini-transcribe and GPT-4o-mini-tts model."""
        super().__init__("GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-tts")
        self.gpt4o_deployment = GPT4O_DEPLOYMENT or "gpt-4o"
        self.tts_deployment = GPT4O_MINI_TTS_DEPLOYMENT or "gpt-4o-mini-tts"
        self.transcribe_deployment = GPT4O_MINI_TRANSCRIBE_DEPLOYMENT or "gpt-4o-mini-transcribe"
        self.minitts_client = None

    async def initialize(self) -> None:
        """
        Initialize the Azure OpenAI clients.

        Raises:
            ModelInitializationError: If client initialization fails
        """
        try:
            # Initialize standard client for GPT-4o and transcribe
            self.client = await create_azure_openai_client(AZURE_OPENAI_API_VERSION, use_minitts_endpoint=False)

            # Initialize separate client for mini-TTS if different endpoint is specified
            if MINITTS_OPENAI_ENDPOINT != AZURE_OPENAI_ENDPOINT or MINITTS_OPENAI_API_KEY != AZURE_OPENAI_API_KEY:
                logger.info("Using separate endpoint for GPT-4o-mini-tts")
                self.minitts_client = await create_azure_openai_client(MINITTS_OPENAI_API_VERSION, use_minitts_endpoint=True)
            else:
                # Use the same client if endpoints are the same
                self.minitts_client = self.client
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-tts model: {e}") from e

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
            # Use the mini-TTS API key if specified
            if MINITTS_OPENAI_API_KEY:
                headers['api-key'] = MINITTS_OPENAI_API_KEY
            else:
                # Use Microsoft Entra ID authentication for TTS
                credential = DefaultAzureCredential()
                token = await credential.get_token("https://cognitiveservices.azure.com/.default")
                headers['Authorization'] = f'Bearer {token.token}'

            return headers
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to get TTS authentication: {e}") from e

    async def generate_response_from_audio(self, audio_data: bytes, text_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for audio input, with GPT-4o-mini-transcribe transcription, GPT-4o processing, and mini-TTS.

        Args:
            audio_data: The input audio data (WAV format)
            text_prompt: Optional text prompt to accompany the audio (will be added to transcript)

        Returns:
            Tuple containing text response, metrics, and audio data

        Raises:
            ModelGenerationError: If response generation fails
        """
        await self.ensure_initialized()
        start_time = time.time()

        try:
            # Step 1: Transcribe audio with GPT-4o-mini-transcribe
            logger.debug(
                f"Transcribing audio with {self.transcribe_deployment}")
            transcribe_start_time = time.time()

            # Create a buffer for the audio file
            buffer = io.BytesIO(audio_data)
            buffer.name = "audio.wav"

            # Call GPT-4o-mini-transcribe for transcription
            transcription_response = await self.client.audio.transcriptions.create(
                model=self.transcribe_deployment,
                file=buffer
            )
            buffer.close()

            transcription = transcription_response.text
            transcribe_complete_time = time.time()
            transcribe_time = transcribe_complete_time - transcribe_start_time

            logger.debug(
                f"Transcription completed in {transcribe_time:.2f}s: {transcription[:50]}...")

            # Combine transcription with any additional text prompt
            if text_prompt:
                full_prompt = f"{text_prompt}\n\nTranscribed audio: {transcription}"
            else:
                full_prompt = f"Respond to this transcribed audio: {transcription}"

            # Step 2: Generate text response with GPT-4o
            logger.debug(
                f"Generating text response with {self.gpt4o_deployment}")
            gpt_start_time = time.time()

            response = await self.client.chat.completions.create(
                model=self.gpt4o_deployment,
                messages=[{"role": "user", "content": full_prompt}],
                stream=False
            )

            text_response = response.choices[0].message.content
            gpt_complete_time = time.time()
            gpt_time = gpt_complete_time - gpt_start_time

            logger.debug(
                f"Text response generated in {gpt_time:.2f}s: {text_response[:50]}...")

            # Step 3: Convert text to speech using GPT-4o-mini-tts
            tts_start_time = time.time()
            logger.debug(
                f"Converting text to speech with {self.tts_deployment}")

            # Get TTS headers with authentication
            headers = await self._get_tts_headers()

            # Prepare TTS API request
            tts_url = f"{MINITTS_OPENAI_ENDPOINT}openai/deployments/{self.tts_deployment}/audio/speech?api-version={MINITTS_OPENAI_API_VERSION}"
            tts_body = {
                "model": self.tts_deployment,
                "input": text_response,
                "voice": "shimmer",
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
            audio_data_output = tts_response.content
            tts_complete_time = time.time()
            tts_generation_time = tts_complete_time - tts_start_time

            logger.debug(f"Audio generated in {tts_generation_time:.2f}s")

            # Calculate audio duration
            audio_duration = get_mp3_duration(
                audio_data_output) if audio_data_output else 0.0

            # Collect metrics with detailed breakdown
            metrics = {
                "model": self.name,
                # Transcription metrics
                "transcribe_time": transcribe_time,

                # Text generation metrics (the core GPT-4o part)
                "text_generation_time": gpt_time,

                # TTS specific metrics
                "tts_time": tts_generation_time,

                # Combined metrics (total pipeline)
                "processing_time": transcribe_time + gpt_time + tts_generation_time,

                # Time until audio would start playing (full pipeline)
                "time_to_audio_start": transcribe_time + gpt_time + tts_generation_time,

                # Actual duration of the generated audio
                "audio_duration": audio_duration,

                # Other metrics
                "token_count": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "audio_size_bytes": len(audio_data_output) if audio_data_output else 0,
                "audio_input_size_bytes": len(audio_data),
                "transcription_length": len(transcription)
            }

            # For consistency with other models that use whisper_time
            metrics["whisper_time"] = transcribe_time

            # Calculate tokens per second based on text generation time only
            metrics["tokens_per_second"] = calculate_tokens_per_second(
                metrics["token_count"], metrics["text_generation_time"]
            )

            logger.info(
                f"Generated response in {metrics['processing_time']:.2f}s (from audio input)")

            return text_response, metrics, audio_data_output

        except Exception as e:
            error_msg = f"Error with {self.name} (audio input): {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error metrics
            error_metrics = create_error_metrics(self.name, e, start_time)
            error_metrics["audio_input_size_bytes"] = len(audio_data)

            return error_msg, error_metrics, None
