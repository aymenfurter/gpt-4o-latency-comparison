import time
import base64
import asyncio
import logging
import io
import wave
from typing import Dict, Any, Tuple, Optional, List

from models.base_model import BaseModel
from utils.exceptions import ModelGenerationError, ModelInitializationError
from utils.audio_utils import convert_pcm_to_wav, get_wav_duration, get_mp3_duration
from utils.client import create_azure_openai_client
from utils.metrics import calculate_tokens_per_second, create_error_metrics
from config import (
    GPT4O_REALTIME_DEPLOYMENT
)

# Configure logger
logger = logging.getLogger(__name__)


class GPT4ORealtimeModel(BaseModel):
    """
    GPT-4o Realtime model implementation.

    Uses the GPT-4o Realtime preview deployment for streaming responses with audio.
    """

    def __init__(self):
        """Initialize the GPT-4o Realtime model."""
        super().__init__("GPT-4o Realtime")
        self.deployment_name = GPT4O_REALTIME_DEPLOYMENT or "gpt-4o-realtime-preview"
        # Use the newer API version required for realtime API
        self.api_version = "2024-10-01-preview"

    async def initialize(self) -> None:
        """
        Initialize the Azure OpenAI client for the realtime API.

        Raises:
            ModelInitializationError: If client initialization fails
        """
        try:
            self.client = await create_azure_openai_client(self.api_version)
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize GPT-4o Realtime model: {e}") from e

    async def generate_response_from_audio(self, audio_data: bytes, text_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for the given audio input with streaming audio output.
        Args:
            audio_data: The input audio data (WAV format)
            text_prompt: Optional text prompt to accompany the audio

        Returns:
            Tuple containing text response, metrics, and audio data

        Raises:
            ModelGenerationError: If response generation fails
        """
        await self.ensure_initialized()
        start_time = time.time()
        first_token_time = None
        response_start_time = None

        # Track metrics
        token_count = 0
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        usage_metrics = {}

        text_chunks: List[str] = []
        audio_chunks: List[bytes] = []

        try:
            logger.info(
                f"Connecting to realtime model: {self.deployment_name}")

            async with self.client.beta.realtime.connect(model=self.deployment_name) as connection:
                # Configure session with required modalities and formats
                await connection.session.update(session={
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16"
                })

                # Record request time
                request_time = time.time()
                logger.debug(
                    f"Request sent at: {request_time - start_time:.3f}s after start")

                # Convert WAV to raw PCM16 if needed
                pcm_audio_data = audio_data

                # Check if audio_data is WAV format and extract PCM data
                if audio_data[:4] == b'RIFF':
                    logger.debug("Detected WAV format, extracting PCM data")
                    try:
                        wav_buffer = io.BytesIO(audio_data)
                        with wave.open(wav_buffer, 'rb') as wav_file:
                            pcm_audio_data = wav_file.readframes(
                                wav_file.getnframes())
                    except Exception as e:
                        logger.error(f"Failed to extract PCM from WAV: {e}")
                        # Fall back to using raw data
                        pcm_audio_data = audio_data

                # Encode audio to base64 for transmission
                encoded_audio = base64.b64encode(
                    pcm_audio_data).decode('utf-8')

                # Create content array with audio and optional text
                content = []

                # Add audio as primary input
                content.append({
                    "type": "input_audio",
                    "audio": encoded_audio
                })

                # Send the user message with audio (and optionally text)
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": content,
                    }
                )

                # Wait for the response to begin
                await connection.response.create()
                response_start_time = time.time()
                logger.debug(
                    f"Response creation started at: {response_start_time - start_time:.3f}s after start")

                # Process streaming response events
                async for event in connection:
                    current_time = time.time()

                    if event.type == "response.text.delta":
                        # Record first token time on first text token
                        if first_token_time is None:
                            first_token_time = current_time
                            logger.debug(
                                f"First token received at: {first_token_time - start_time:.3f}s after start")

                        # Count tokens and collect text
                        token_count += 1
                        text_chunks.append(event.delta)

                    elif event.type == "response.audio.delta":
                        # Record first audio time if no text was received
                        if first_token_time is None:
                            first_token_time = current_time
                            logger.debug(
                                f"First audio received at: {first_token_time - start_time:.3f}s after start")

                        # Collect audio chunks
                        audio_data = base64.b64decode(event.delta)
                        audio_chunks.append(audio_data)

                    elif event.type == "rate_limits.updated":
                        # Log rate limit information if available
                        if hasattr(event, 'rate_limits'):
                            logger.debug(f"Rate limits: {event.rate_limits}")

                    elif event.type == "response.done":
                        # Extract usage information from the completed response
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            usage = event.response.usage
                            logger.debug(f"Response usage metrics: {usage}")

                            # Capture token metrics from the final usage statistics
                            if hasattr(usage, 'total_tokens'):
                                total_tokens = usage.total_tokens
                            if hasattr(usage, 'input_tokens'):
                                input_tokens = usage.input_tokens
                            if hasattr(usage, 'output_tokens'):
                                output_tokens = usage.output_tokens

                            # Get detailed token information if available
                            if hasattr(usage, 'output_token_details'):
                                if hasattr(usage.output_token_details, 'text_tokens'):
                                    usage_metrics['text_tokens'] = usage.output_token_details.text_tokens
                                if hasattr(usage.output_token_details, 'audio_tokens'):
                                    usage_metrics['audio_tokens'] = usage.output_token_details.audio_tokens
                        break

            # Combine text and audio data
            text_response = "".join(text_chunks)
            raw_audio_data = b"".join(audio_chunks)

            # Process audio data
            audio_output = None
            audio_duration = 0.0
            if raw_audio_data:
                audio_output = convert_pcm_to_wav(raw_audio_data)
                audio_duration = get_wav_duration(
                    audio_output) if audio_output else 0.0

            # Ensure first_token_time is set
            if first_token_time is None:
                first_token_time = response_start_time or time.time()
                logger.warning(
                    "No tokens received, using response start time for first token latency")

            # Use output_tokens if no tokens were counted in stream
            if token_count == 0 and output_tokens > 0:
                token_count = output_tokens
                logger.debug(
                    f"Using output_tokens ({output_tokens}) from usage metrics for token_count")

            # Calculate metrics
            end_time = time.time()
            metrics = {
                "model": self.name,
                "processing_time": float(end_time - start_time),
                "time_to_first_token": float(first_token_time - start_time),
                "time_to_audio_start": float(first_token_time - start_time),
                "audio_duration": float(audio_duration),
                "token_count": int(token_count or 0),
                "total_tokens": int(total_tokens or 0),
                "input_tokens": int(input_tokens or 0),
                "output_tokens": int(output_tokens or 0),
                "audio_chunk_count": int(len(audio_chunks)),
                "audio_size_bytes": int(len(audio_output) if audio_output else 0),
                "audio_input_size_bytes": int(len(audio_data))
            }

            # Add usage metrics and calculate tokens_per_second
            metrics.update(usage_metrics)
            if token_count > 0:
                metrics["tokens_per_second"] = float(calculate_tokens_per_second(
                    token_count, metrics["processing_time"]
                ))
            else:
                metrics["tokens_per_second"] = 0.0

            logger.info(
                f"Generated realtime response in {metrics['processing_time']:.2f}s (from audio input)")

            return text_response, metrics, audio_output

        except Exception as e:
            error_msg = f"Error with GPT-4o Realtime: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Ensure first_token_time is set for metrics
            if first_token_time is None:
                first_token_time = start_time

            # Return error metrics with numeric values to avoid type errors
            error_metrics = create_error_metrics(self.name, e, start_time)
            error_metrics.update({
                "time_to_audio_start": float(first_token_time - start_time),
                "audio_chunk_count": int(len(audio_chunks)),
                "audio_size_bytes": int(sum(len(chunk) for chunk in audio_chunks) if audio_chunks else 0),
                "audio_input_size_bytes": int(len(audio_data))
            })

            return error_msg, error_metrics, None
