import time
import requests
import json
import base64
import io
import tempfile
import os
from typing import Dict, Any, Tuple, Optional

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from models.base_model import BaseModel
from config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION,
    GPT4O_DEPLOYMENT, TTS_DEPLOYMENT
)


class GPT4OWhisperModel(BaseModel):
    """GPT-4o with Whisper TTS model implementation."""

    def __init__(self):
        super().__init__("GPT-4o + Whisper TTS")
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

    def _get_mp3_duration(self, mp3_data: bytes) -> float:
        """Get the duration of an MP3 audio file in seconds."""
        try:
            # Use ffprobe directly - the most reliable method for MP3
            import subprocess
            import tempfile
            import os

            # Write to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp:
                temp_path = temp.name
                temp.write(mp3_data)

            # Try using ffprobe to get duration (available in most environments)
            try:
                cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'default=noprint_wrappers=1:nokey=1', temp_path]
                result = subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

                if result.returncode == 0:
                    duration = float(result.stdout.strip())
                    print(
                        f"MP3 duration detected with ffprobe: {duration:.2f} seconds")

                    # Clean up
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

                    return duration
            except:
                # If ffprobe fails or isn't available, try pydub
                pass

            # Try with pydub (needs ffmpeg installed)
            try:
                from pydub import AudioSegment
                audio = AudioSegment.from_mp3(temp_path)
                duration = len(audio) / 1000.0  # pydub uses milliseconds
                print(
                    f"MP3 duration detected with pydub: {duration:.2f} seconds")

                # Clean up
                try:
                    os.unlink(temp_path)
                except:
                    pass

                return duration
            except Exception as e:
                print(f"Error determining MP3 duration with pydub: {e}")

                # Clean up if we're still here
                try:
                    os.unlink(temp_path)
                except:
                    pass
        except Exception as e:
            print(f"Error setting up duration detection: {e}")

        # Final fallback: Calculate based on typical MP3 bitrate
        # For speech at 64kbps, we'd have about 8KB per second
        try:
            # Get filesize
            size_kb = len(mp3_data) / 1024

            # Typical speech MP3 is 8-16KB per second depending on quality
            bitrate_kb_per_sec = 8  # Assuming low bitrate for speech

            # Calculate duration
            duration = size_kb / bitrate_kb_per_sec

            # Apply reasonable constraints
            duration = min(180, max(5, duration))  # Between 5 and 180 seconds

            print(
                f"Calculated MP3 duration from file size: {duration:.2f} seconds")
            return duration
        except Exception as e:
            print(f"Error in fallback duration calculation: {e}")

        # Absolute last resort - return a reasonable default
        return 60.0  # Assume a default 1-minute duration

    async def generate_response(self, prompt: str) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        if not self.client:
            await self._initialize_client()

        start_time = time.time()

        # Step 1: Generate text response with GPT-4o
        response = await self.client.chat.completions.create(
            model=GPT4O_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )

        text_response = response.choices[0].message.content
        text_complete_time = time.time()
        text_generation_time = text_complete_time - start_time

        # Step 2: Convert text to speech
        tts_start_time = time.time()

        # Prepare headers for TTS request
        headers = {'Content-Type': 'application/json'}

        if AZURE_OPENAI_API_KEY:
            headers['api-key'] = AZURE_OPENAI_API_KEY
        else:
            # Use Microsoft Entra ID authentication for TTS
            credential = DefaultAzureCredential()
            token = await credential.get_token("https://cognitiveservices.azure.com/.default")
            headers['Authorization'] = f'Bearer {token.token}'

        tts_url = f"{AZURE_OPENAI_ENDPOINT}openai/deployments/{TTS_DEPLOYMENT}/audio/speech?api-version={AZURE_OPENAI_API_VERSION}"
        tts_body = {
            "input": text_response,
            "voice": "nova",
            "response_format": "mp3"
        }

        tts_response = requests.post(
            tts_url, headers=headers, data=json.dumps(tts_body))
        audio_data = tts_response.content

        tts_complete_time = time.time()
        tts_generation_time = tts_complete_time - tts_start_time

        # Get actual audio duration from the MP3 file
        audio_duration = self._get_mp3_duration(
            audio_data) if audio_data else 0.0

        # Collect metrics
        metrics = {
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

        # Calculate tokens per second based on the text generation time (not the total time)
        # This gives a fair comparison with other models
        if metrics["token_count"] > 0 and metrics["text_generation_time"] > 0:
            metrics["tokens_per_second"] = metrics["token_count"] / \
                metrics["text_generation_time"]

        return text_response, metrics, audio_data
