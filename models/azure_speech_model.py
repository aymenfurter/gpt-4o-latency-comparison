import time
import io
import logging
import tempfile
import os
import asyncio
from typing import Dict, Any, Tuple, Optional, List

import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI

from models.base_model import BaseModel
from utils.exceptions import ModelGenerationError, ModelInitializationError
from utils.audio_utils import get_wav_duration, get_mp3_duration
from utils.metrics import calculate_tokens_per_second, create_error_metrics
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    GPT4O_DEPLOYMENT,
    SPEECH_KEY,
    SPEECH_REGION,
    SPEECH_RECOGNITION_LANGUAGE,
    SPEECH_SYNTHESIS_VOICE
)

logger = logging.getLogger(__name__)


class AzureSpeechModel(BaseModel):
    """
    Azure Speech + OpenAI model implementation.

    Uses Azure Speech Services for STT and TTS, and Azure OpenAI for text generation.
    """

    def __init__(self):
        """Initialize the Azure Speech + OpenAI model."""
        super().__init__("Azure Speech + OpenAI")
        self.gpt4o_deployment = GPT4O_DEPLOYMENT or "gpt-4o"
        self.tts_sentence_end = [".", "!", "?", ";", "。", "！", "？", "；", "\n"]

    async def initialize(self) -> None:
        """
        Initialize the speech recognizer, synthesizer, and OpenAI client.

        Raises:
            ModelInitializationError: If client initialization fails
        """
        try:
            if not SPEECH_KEY or not SPEECH_REGION:
                raise ModelInitializationError(
                    "Missing Speech Key or Region. Set SPEECH_KEY and SPEECH_REGION environment variables.")

            # Initialize the OpenAI client
            self.client = AzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_key=AZURE_OPENAI_API_KEY,
                api_version=AZURE_OPENAI_API_VERSION
            )

            # Initialize Speech SDK components
            self.speech_config = speechsdk.SpeechConfig(
                subscription=SPEECH_KEY, region=SPEECH_REGION)

            # Configure speech recognition settings
            self.speech_config.speech_recognition_language = SPEECH_RECOGNITION_LANGUAGE

            # Configure speech synthesis settings
            self.speech_config.speech_synthesis_voice_name = SPEECH_SYNTHESIS_VOICE

            logger.info(
                f"Initialized Azure Speech + OpenAI with voice: {SPEECH_SYNTHESIS_VOICE}")

        except Exception as e:
            raise ModelInitializationError(
                f"Failed to initialize Azure Speech model: {e}") from e

    async def generate_response_from_audio(self, audio_data: bytes, text_prompt: Optional[str] = None) -> Tuple[str, Dict[str, Any], Optional[bytes]]:
        """
        Generate a response for audio input using Speech Services for STT and TTS.

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

        try:
            # Step 1: Create a temporary file to store the input audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_filename = temp_file.name
                temp_file.write(audio_data)

            stt_start_time = time.time()

            # Step 2: Configure speech recognition from the audio file
            audio_input = speechsdk.audio.AudioConfig(filename=temp_filename)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config,
                audio_config=audio_input
            )

            # Recognize speech from the audio file
            logger.debug("Starting speech recognition")
            result = speech_recognizer.recognize_once_async().get()

            # Process speech recognition result
            transcription = ""
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                transcription = result.text
                logger.debug(f"Recognized speech: {transcription}")
            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech could be recognized")
                transcription = "No speech could be recognized."
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                logger.error(
                    f"Speech recognition canceled: {cancellation.reason}")
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    raise ModelGenerationError(
                        f"STT error: {cancellation.error_details}")

            stt_complete_time = time.time()
            stt_time = stt_complete_time - stt_start_time

            # Clean up the input audio file
            try:
                os.unlink(temp_filename)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

            # Step 3: Process the transcription with OpenAI
            # Combine transcription with any additional text prompt
            if text_prompt:
                full_prompt = f"{text_prompt}\n\nTranscribed audio: {transcription}"
            else:
                full_prompt = f"Respond to this transcribed audio: {transcription}"

            gpt_start_time = time.time()
            logger.debug(f"Generating text response for: {full_prompt}")

            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.gpt4o_deployment,
                messages=[{"role": "user", "content": full_prompt}],
                max_tokens=1000,
                stream=False
            )

            text_response = response.choices[0].message.content
            gpt_complete_time = time.time()
            gpt_time = gpt_complete_time - gpt_start_time

            # Step 4: Convert the response to speech
            tts_start_time = time.time()
            logger.debug("Converting text to speech")

            # Prepare file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as out_file:
                output_filename = out_file.name

            # Configure audio output
            audio_output_config = speechsdk.audio.AudioOutputConfig(
                filename=output_filename)

            # Create synthesizer
            speech_synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_output_config
            )

            # Synthesize speech
            result = speech_synthesizer.speak_text_async(text_response).get()

            # Check synthesis result
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.debug(f"Speech synthesized to file: {output_filename}")
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation_details = result.cancellation_details
                logger.error(
                    f"Speech synthesis canceled: {cancellation_details.reason}")
                if cancellation_details.reason == speechsdk.CancellationReason.Error:
                    raise ModelGenerationError(
                        f"TTS error: {cancellation_details.error_details}")

            # Read the synthesized audio data
            with open(output_filename, "rb") as audio_file:
                audio_data_output = audio_file.read()

            # Clean up the output file
            try:
                os.unlink(output_filename)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")

            tts_complete_time = time.time()
            tts_time = tts_complete_time - tts_start_time

            # Calculate audio duration
            audio_duration = get_wav_duration(
                audio_data_output) if audio_data_output else 0.0

            # Collect metrics with detailed breakdown
            metrics = {
                "model": self.name,
                # STT transcription metrics
                "transcribe_time": stt_time,  # Using whisper_time for consistency with other models

                # Text generation metrics
                "text_generation_time": gpt_time,

                # TTS specific metrics
                "tts_time": tts_time,

                # Combined metrics
                "processing_time": stt_time + gpt_time + tts_time,

                # Time until audio would start playing
                "time_to_audio_start": stt_time + gpt_time + tts_time,

                # Actual duration of the generated audio
                "audio_duration": audio_duration,

                # Other metrics
                "token_count": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "audio_size_bytes": len(audio_data_output) if audio_data_output else 0,
                "audio_input_size_bytes": len(audio_data),
                "transcription_length": len(transcription)
            }

            # Calculate tokens per second based on text generation time
            metrics["tokens_per_second"] = calculate_tokens_per_second(
                metrics["token_count"], metrics["text_generation_time"]
            )

            logger.info(
                f"Generated Azure Speech response in {metrics['processing_time']:.2f}s (from audio input)")

            return text_response, metrics, audio_data_output

        except Exception as e:
            error_msg = f"Error with Azure Speech model: {str(e)}"
            logger.error(error_msg, exc_info=True)

            # Return error metrics
            error_metrics = create_error_metrics(self.name, e, start_time)
            error_metrics["audio_input_size_bytes"] = len(audio_data)

            return error_msg, error_metrics, None
