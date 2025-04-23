import time
import base64
import wave
import io
from typing import Dict, Any, Tuple, Optional

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from models.base_model import BaseModel
from config import AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, GPT4O_AUDIO_DEPLOYMENT


class GPT4OAudioModel(BaseModel):
    """GPT-4o Audio model implementation."""

    def __init__(self):
        super().__init__("GPT-4o Audio Preview")
        self.client = None

    async def _initialize_client(self):
        if AZURE_OPENAI_API_KEY:
            self.client = AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=AZURE_OPENAI_API_VERSION
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
                api_version=AZURE_OPENAI_API_VERSION
            )

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

            # Second approach: Try using pydub
            try:
                import tempfile
                import os
                from pydub import AudioSegment

                # Write to a temporary file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_file_path = temp_file.name
                    temp_file.write(wav_data)

                # Use pydub to get duration
                audio = AudioSegment.from_wav(temp_file_path)
                duration = len(audio) / 1000.0  # pydub uses milliseconds

                # Clean up the temporary file
                try:
                    os.unlink(temp_file_path)
                except:
                    pass

                print(
                    f"WAV duration detected with pydub: {duration:.2f} seconds")
                return duration

            except Exception as inner_e:
                print(f"Error determining WAV duration with pydub: {inner_e}")

            # Final fallback: Estimate from file size
            # WAV files are typically around 172KB per second for 16-bit 44.1kHz stereo
            # But for speech, we can assume mono and maybe lower sample rate ~ 86KB/s
            estimated_duration = len(wav_data) / (86 * 1024)

            # Apply reasonable constraints
            if estimated_duration > 180:
                estimated_duration = min(60, len(wav_data) / (172 * 1024))

            print(
                f"Estimated WAV duration based on file size: {estimated_duration:.2f} seconds")
            return estimated_duration

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        if not self.client:
            await self._initialize_client()

        start_time = time.time()

        # Make the audio chat completions request
        response = await self.client.chat.completions.create(
            model=GPT4O_AUDIO_DEPLOYMENT,
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
        first_token_time = time.time()
        processing_time = first_token_time - start_time

        text_response = response.choices[0].message.audio.transcript
        audio_data = base64.b64decode(response.choices[0].message.audio.data)

        # Calculate actual audio duration from the WAV file
        audio_duration = self._get_wav_duration(
            audio_data) if audio_data else 0.0

        metrics = {
            "text_generation_time": processing_time,
            "time_to_audio_start": processing_time,  # Time until audio is ready to play
            # Total processing time (same for this model)
            "processing_time": processing_time,
            "audio_duration": audio_duration,         # Actual audio playback duration
            "token_count": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
            "audio_size_bytes": len(audio_data)
        }

        if metrics["token_count"] > 0 and metrics["processing_time"] > 0:
            metrics["tokens_per_second"] = metrics["token_count"] / \
                metrics["processing_time"]

        return text_response, metrics, audio_data
