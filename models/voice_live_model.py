import time
import base64
import asyncio
import logging
import json
import uuid
import io
import wave
from typing import Dict, Any, Tuple, Optional, List

from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import WebSocketException

from models.base_model import BaseModel
from utils.exceptions import ModelGenerationError, ModelInitializationError
from utils.audio_utils import convert_pcm_to_wav, get_wav_duration
from utils.metrics import create_error_metrics
from config import (
    VOICE_LIVE_ENDPOINT,
    VOICE_LIVE_API_VERSION,
    VOICE_LIVE_MODEL_NAME,
    VOICE_LIVE_KEY
)
from azure.identity.aio import DefaultAzureCredential

# Configure logger
logger = logging.getLogger(__name__)


class VoiceLiveModel(BaseModel):
    """
    Azure Voice Live API model implementation.

    Uses a WebSocket connection for real-time, low-latency speech-to-speech interaction.
    """

    def __init__(self):
        """Initialize the Azure Voice Live model."""
        super().__init__("Azure Voice Live")
        self.endpoint = VOICE_LIVE_ENDPOINT
        self.api_version = VOICE_LIVE_API_VERSION
        self.model_name = VOICE_LIVE_MODEL_NAME
        self.api_key = VOICE_LIVE_KEY

    async def initialize(self) -> None:
        """
        Initialize the Azure credential for authentication.
        """
        try:
            if not self.endpoint:
                raise ModelInitializationError(
                    "VOICE_LIVE_ENDPOINT is not configured.")
            logger.info(
                f"Initialized Azure Voice Live model for endpoint: {self.endpoint}")
        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize Azure Voice Live model: {e}") from e

    async def generate_response_from_audio(self, audio_data: bytes, text_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        await self.ensure_initialized()

        max_retries = 5
        retry_count = 0
        last_error = None

        while retry_count < max_retries:
            try:
                url = f"{self.endpoint.rstrip('/')}/voice-live/realtime?api-version={self.api_version}&model={self.model_name}"
                url = url.replace("https://", "wss://")
                headers = {"api-key": f"{self.api_key}"}

                async with ws_connect(url, additional_headers=headers) as websocket:
                    session_update = {
                        "type": "session.update",
                        "session": {
                            "instructions": "Answer concisely",
                            "voice": {"name": "en-US-AvaNeural", "type": "azure-standard"},
                        }
                    }
                    await websocket.send(json.dumps(session_update))
                    start_time = time.time()

                    await self._send_audio_chunks(websocket, audio_data)
                    text_response, metrics, audio_output = await self._receive_response(websocket, start_time)

                    if not audio_output or len(audio_output) < 1000:
                        raise ModelGenerationError(
                            "No valid audio response received")

                    end_time = time.time()
                    metrics["model"] = self.name
                    metrics["processing_time"] = end_time - start_time
                    metrics["audio_input_size_bytes"] = len(audio_data)
                    metrics["retry_count"] = retry_count

                    logger.info(
                        f"Generated Voice Live response in {metrics['processing_time']:.2f}s after {retry_count} retries")
                    return text_response, metrics, audio_output

            except (WebSocketException, ModelGenerationError, Exception) as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(
                        f"Attempt {retry_count} failed, retrying... Error: {str(e)}")
                    await asyncio.sleep(1)  # Wait 1 second before retrying
                else:
                    logger.error(
                        f"All {max_retries} attempts failed. Last error: {str(e)}")
                    error_msg = f"Voice Live generation failed after {max_retries} attempts: {str(e)}"
                    error_metrics = create_error_metrics(
                        self.name, e, start_time)
                    error_metrics["retry_count"] = retry_count
                    return error_msg, error_metrics, None

    async def _send_audio_chunks(self, websocket, audio_data: bytes):
        """Sends the input audio data in chunks over the WebSocket."""
        try:
            with wave.open(io.BytesIO(audio_data), 'rb') as wav_file:
                if wav_file.getsampwidth() != 2 or wav_file.getnchannels() != 1:
                    raise ModelGenerationError(
                        "Voice Live requires 16-bit mono WAV input.")
                sample_rate = wav_file.getframerate()
                pcm_data = wav_file.readframes(wav_file.getnframes())
        except wave.Error as e:
            raise ModelGenerationError(f"Invalid WAV file provided: {e}")

        chunk_size = int(sample_rate * 0.02) * 2  # 20ms of 16-bit audio
        for i in range(0, len(pcm_data), chunk_size):
            chunk = pcm_data[i:i + chunk_size]
            encoded_audio = base64.b64encode(chunk).decode('utf-8')
            message = {
                "type": "input_audio_buffer.append",
                "audio": encoded_audio
            }
            await websocket.send(json.dumps(message))
            await asyncio.sleep(0.01)

        await websocket.send(json.dumps({"type": "input_audio_buffer.commit"}))
        await websocket.send(json.dumps({"type": "response.create"}))

    async def _receive_response(self, websocket, start_time: float) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """Receives and processes events from the WebSocket."""
        first_audio_time = None
        text_chunks = []
        audio_chunks = []
        metrics = {}
        response_complete = False
        audio_stream_complete = False

        timeout = 15.0

        async def receive_messages():
            nonlocal first_audio_time, response_complete, audio_stream_complete
            async for message in websocket:
                event = json.loads(message)
                event_type = event.get("type")

                if event_type == "response.audio.delta":
                    if first_audio_time is None:
                        first_audio_time = time.time()

                    delta_audio = base64.b64decode(event.get("delta", ""))
                    audio_chunks.append(delta_audio)

                elif event_type == "response.audio_transcript.delta":
                    text_chunks.append(event.get("delta", ""))

                elif event_type == "error":
                    error_details = event.get("error", {})
                    raise ModelGenerationError(
                        f"Voice Live API Error: {error_details.get('message', 'Unknown error')}")

                elif event_type == "response.done":
                    response_obj = event.get("response", {})
                    usage = response_obj.get("usage", {})
                    metrics.update({
                        "token_count": usage.get("output_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    })
                    response_complete = True

                elif event_type == "response.audio.done":
                    logger.debug("Audio streaming completed")
                    audio_stream_complete = True

                if response_complete and audio_stream_complete:
                    break

        try:
            await asyncio.wait_for(receive_messages(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"WebSocket response timed out after {timeout}s")
            if not response_complete:
                metrics = {"token_count": 0, "total_tokens": 0}

        text_response = "".join(text_chunks)
        raw_audio_data = b"".join(audio_chunks)

        audio_output = None
        audio_duration = 0.0
        if raw_audio_data:
            audio_output = convert_pcm_to_wav(
                raw_audio_data, sample_rate=24000)
            audio_duration = get_wav_duration(audio_output)

        metrics["time_to_audio_start"] = (
            first_audio_time - start_time) if first_audio_time else 0
        metrics["audio_duration"] = audio_duration
        metrics["audio_size_bytes"] = len(audio_output) if audio_output else 0

        logger.debug(
            f"Received {len(text_chunks)} text chunks and {len(audio_chunks)} audio chunks")

        return text_response, metrics, audio_output
