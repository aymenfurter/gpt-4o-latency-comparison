import time
import base64
import asyncio
import wave
import io
import struct
from typing import Dict, Any, Tuple, Optional, List

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from models.base_model import BaseModel
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, GPT4O_REALTIME_DEPLOYMENT


class GPT4ORealtimeModel(BaseModel):
    """GPT-4o Realtime model implementation."""

    def __init__(self):
        super().__init__("GPT-4o Realtime")
        self.client = None

    async def _initialize_client(self):
        # Use the newer API version required for realtime API
        api_version = "2024-10-01-preview"

        if AZURE_OPENAI_API_KEY:
            self.client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=api_version
            )
        else:
            # Use Microsoft Entra ID authentication
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            self.client = AsyncAzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_ad_token_provider=token_provider,
                api_version=api_version
            )

    def _convert_pcm_to_wav(self, pcm_data: bytes) -> bytes:
        """
        Convert raw PCM16 audio data to WAV format for playback.

        PCM16 is 16-bit linear PCM data, which needs WAV headers to be playable.
        """
        # Create an in-memory buffer
        wav_buffer = io.BytesIO()

        # PCM16 format parameters
        channels = 1  # Mono
        sample_width = 2  # 16 bits = 2 bytes
        # Common for speech audio (Azure uses 24kHz for realtime API)
        sample_rate = 24000

        # Create WAV file with proper headers
        with wave.open(wav_buffer, 'wb') as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(pcm_data)

        # Get the complete WAV data
        wav_data = wav_buffer.getvalue()
        return wav_data

    def _get_wav_duration(self, wav_data: bytes) -> float:
        """Get the duration of a WAV audio file in seconds."""
        try:
            # First approach: Use wave module
            wav_file = wave.open(io.BytesIO(wav_data), 'rb')
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)

            # Sanity check - if unrealistic, try other methods
            if duration > 300 or duration < 1:
                raise ValueError("Unrealistic duration detected")

            print(
                f"WAV duration detected with wave module: {duration:.2f} seconds")
            return duration
        except Exception as e:
            print(f"Error determining WAV duration with wave: {e}")

            # Second approach: Try using audio length and sample rate from header
            try:
                # WAV header structure analysis
                # First 44 bytes contain the header
                if len(wav_data) > 44:
                    # Sample rate is at offset 24-27 (little-endian)
                    sample_rate = int.from_bytes(
                        wav_data[24:28], byteorder='little')

                    # Data length is near offset 40-43
                    # First find the 'data' chunk
                    for i in range(36, min(100, len(wav_data)-8)):
                        if wav_data[i:i+4] == b'data':
                            # Data size is in the next 4 bytes
                            data_size = int.from_bytes(
                                wav_data[i+4:i+8], byteorder='little')

                            # Calculate duration - adjust for bit depth (16-bit = 2 bytes per sample)
                            # For mono (1 channel)
                            bit_depth = 16  # Typical for PCM
                            channels = 1  # Mono for speech
                            bytes_per_sample = bit_depth // 8 * channels

                            if bytes_per_sample > 0 and sample_rate > 0:
                                duration = data_size / \
                                    (bytes_per_sample * sample_rate)
                                print(
                                    f"WAV duration from header analysis: {duration:.2f} seconds")
                                return duration
            except Exception as header_e:
                print(f"Error analyzing WAV header: {header_e}")

            # Final fallback: Estimate from file size
            # For PCM16 mono at 24kHz (typical for Azure), we have 48KB/s
            bytes_per_second = 24000 * 2  # 24kHz * 2 bytes per sample
            estimated_duration = (len(wav_data) - 44) / \
                bytes_per_second  # Subtract header size

            # Apply reasonable constraints
            if estimated_duration > 180 or estimated_duration < 1:
                estimated_duration = min(
                    60, max(5, len(wav_data) / 50000))  # Rough estimate

            print(
                f"Estimated WAV duration based on file size: {estimated_duration:.2f} seconds")
            return estimated_duration

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        await self._initialize_client()

        start_time = time.time()
        first_token_time = None
        first_token_received = False
        response_start_time = None

        # Track token metrics
        token_count = 0
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        usage_metrics = {}

        text_chunks: List[str] = []
        audio_chunks: List[bytes] = []

        try:
            model_name = GPT4O_REALTIME_DEPLOYMENT or "gpt-4o-realtime-preview"

            print(f"Connecting to realtime model: {model_name}")

            async with self.client.beta.realtime.connect(
                model=model_name,
            ) as connection:
                # Enable text and audio modalities and specify audio formats
                await connection.session.update(session={
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "output_audio_format": "pcm16"
                })

                # Record when we send the request
                request_time = time.time()
                print(
                    f"Request sent at: {request_time - start_time:.3f}s after start")

                # Send the user message
                await connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": prompt}],
                    }
                )

                # Wait for the response
                await connection.response.create()
                response_start_time = time.time()
                print(
                    f"Response creation started at: {response_start_time - start_time:.3f}s after start")

                # Process streaming response
                async for event in connection:
                    current_time = time.time()

                    # Check event type for metrics and content
                    if event.type == "response.text.delta":
                        # Set first token time only on the first text token
                        if not first_token_received:
                            first_token_time = current_time
                            first_token_received = True
                            print(
                                f"First token received at: {first_token_time - start_time:.3f}s after start")

                        # Count tokens (each delta is a token in realtime API)
                        token_count += 1
                        text_chunks.append(event.delta)

                    elif event.type == "response.audio.delta":
                        audio_data = base64.b64decode(event.delta)
                        audio_chunks.append(audio_data)

                    elif event.type == "rate_limits.updated":
                        # Extract rate limit information if available
                        if hasattr(event, 'rate_limits'):
                            print(f"Rate limits: {event.rate_limits}")

                    elif event.type == "response.done":
                        # Extract usage information from the completed response
                        if hasattr(event, 'response') and hasattr(event.response, 'usage'):
                            print(
                                f"Response usage metrics: {event.response.usage}")
                            usage = event.response.usage

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

            text_response = "".join(text_chunks)
            raw_audio_data = b"".join(audio_chunks)

            # Convert the PCM16 audio data to WAV format for playback
            audio_data = self._convert_pcm_to_wav(
                raw_audio_data) if raw_audio_data else None

            # Calculate actual audio duration
            audio_duration = self._get_wav_duration(
                audio_data) if audio_data else 0.0

            end_time = time.time()

            # Ensure first_token_time is never None
            if first_token_time is None:
                # If no tokens were received, use the response start time or end time
                first_token_time = response_start_time or end_time
                print(
                    "Warning: No tokens received, using response start time for first token latency")

            # If we didn't get the token count from the delta events, use the output tokens from usage
            if token_count == 0 and output_tokens > 0:
                token_count = output_tokens
                print(
                    f"Using output_tokens ({output_tokens}) from usage metrics for token_count")

            # Calculate metrics
            metrics = {
                "model": self.name,
                "processing_time": end_time - start_time,
                "time_to_audio_start": first_token_time - start_time,
                "audio_duration": audio_duration,
                "token_count": token_count,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "audio_chunk_count": len(audio_chunks),
                "audio_size_bytes": len(audio_data) if audio_data else 0
            }

            # Add additional metrics from usage if available
            metrics.update(usage_metrics)

            # Calculate tokens per second if we have tokens
            if metrics["token_count"] > 0 and metrics["processing_time"] > 0:
                metrics["tokens_per_second"] = metrics["token_count"] / \
                    metrics["processing_time"]
            else:
                metrics["tokens_per_second"] = 0

            print(f"Realtime metrics: {metrics}")

            return text_response, metrics, audio_data

        except Exception as e:
            # Add more detailed error information
            error_msg = f"Error connecting to GPT-4o Realtime: {str(e)}"
            print(error_msg)

            # Ensure first_token_time is never None for metrics calculation
            if first_token_time is None:
                first_token_time = start_time  # Will result in zero latency, but won't crash

            # Return error details in the response
            return error_msg, {
                "model": self.name,
                "error": str(e),
                "total_duration": time.time() - start_time,
                "first_token_latency": first_token_time - start_time,
                "token_count": token_count,
                "total_tokens": total_tokens,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "audio_chunk_count": len(audio_chunks),
                "audio_size_bytes": len(audio_chunks) and sum(len(chunk) for chunk in audio_chunks) or 0,
                "tokens_per_second": 0
            }, None
