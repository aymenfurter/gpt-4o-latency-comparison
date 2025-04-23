"""Utility functions for audio processing and analysis."""

import wave
import io
import os
import tempfile
import subprocess
from typing import Optional


def convert_pcm_to_wav(pcm_data: bytes, sample_rate: int = 24000) -> bytes:
    """
    Convert raw PCM16 audio data to WAV format for playback.

    Args:
        pcm_data: Raw PCM16 audio data
        sample_rate: Sample rate in Hz (default: 24000 for Azure)

    Returns:
        WAV formatted audio data
    """
    wav_buffer = io.BytesIO()
    channels = 1  # Mono
    sample_width = 2  # 16 bits = 2 bytes

    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_data)

    return wav_buffer.getvalue()


def get_wav_duration(wav_data: bytes) -> float:
    """
    Get the duration of a WAV audio file in seconds.

    Args:
        wav_data: WAV audio data in bytes

    Returns:
        Duration in seconds
    """
    try:
        # First approach: Use wave module
        with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
            frames = wav_file.getnframes()
            rate = wav_file.getframerate()
            duration = frames / float(rate)

        # Sanity check
        if 1 <= duration <= 300:
            return duration
    except Exception as e:
        print(f"Wave module error: {e}")

    try:
        # Second approach: Try with pydub
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp:
            temp_path = temp.name
            temp.write(wav_data)

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_wav(temp_path)
            duration = len(audio) / 1000.0  # pydub uses milliseconds
            return duration
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    except Exception as e:
        print(f"Pydub error: {e}")

    # Fallback: Estimate from file size
    # Typical WAV is ~86KB/s for mono speech
    estimated_duration = len(wav_data) / (86 * 1024)
    return min(180, max(5, estimated_duration))


def get_mp3_duration(mp3_data: bytes) -> float:
    """
    Get the duration of an MP3 audio file in seconds.

    Args:
        mp3_data: MP3 audio data in bytes

    Returns:
        Duration in seconds
    """
    # Use ffprobe if available
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp:
            temp_path = temp.name
            temp.write(mp3_data)

        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                   '-of', 'default=noprint_wrappers=1:nokey=1', temp_path]
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

            if result.returncode == 0:
                return float(result.stdout.strip())
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    except Exception as e:
        print(f"ffprobe error: {e}")

    # Try with pydub
    try:
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp:
            temp_path = temp.name
            temp.write(mp3_data)

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(temp_path)
            return len(audio) / 1000.0
        finally:
            try:
                os.unlink(temp_path)
            except:
                pass
    except Exception as e:
        print(f"Pydub error: {e}")

    # Fallback: Calculate based on typical MP3 bitrate
    size_kb = len(mp3_data) / 1024
    bitrate_kb_per_sec = 8  # Assuming low bitrate for speech
    estimated_duration = size_kb / bitrate_kb_per_sec

    return min(180, max(5, estimated_duration))
