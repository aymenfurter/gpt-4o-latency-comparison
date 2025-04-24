"""Configuration settings for the GPT-4o latency comparison project."""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables from .env file if present
load_dotenv()

# =========================================
# Azure OpenAI Configuration
# =========================================
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2025-01-01-preview"
)

# Validate required configuration
if not AZURE_OPENAI_ENDPOINT:
    logging.warning("AZURE_OPENAI_ENDPOINT environment variable is not set")

# =========================================
# Azure Speech Services Configuration
# =========================================
SPEECH_KEY = os.getenv("SPEECH_KEY")
SPEECH_REGION = os.getenv("SPEECH_REGION")
SPEECH_RECOGNITION_LANGUAGE = os.getenv("SPEECH_RECOGNITION_LANGUAGE", "en-US")
SPEECH_SYNTHESIS_VOICE = os.getenv(
    "SPEECH_SYNTHESIS_VOICE", "en-US-JennyMultilingualNeural")

# =========================================
# Model Deployment Names
# =========================================
GPT4O_DEPLOYMENT = os.getenv("GPT4O_DEPLOYMENT", "gpt-4o")
GPT4O_REALTIME_DEPLOYMENT = os.getenv(
    "GPT4O_REALTIME_DEPLOYMENT", "gpt-4o-realtime-preview"
)
GPT4O_AUDIO_DEPLOYMENT = os.getenv(
    "GPT4O_AUDIO_DEPLOYMENT", "gpt-4o-audio-preview"
)
GPT41_MINI_DEPLOYMENT = os.getenv("GPT41_MINI_DEPLOYMENT", "gpt-4.1-mini")
WHISPER_DEPLOYMENT = os.getenv("WHISPER_DEPLOYMENT", "whisper")
TTS_DEPLOYMENT = os.getenv("TTS_DEPLOYMENT", "tts")

# =========================================
# Benchmark Settings
# =========================================
# Define a concise test prompt for benchmark comparisons
DEFAULT_PROMPT = "Translate the sentence 'Where is the trainstation?' to Spanish. Respond with only a single sentence."

# Number of iterations to run for each benchmark
try:
    BENCHMARK_ITERATIONS = int(os.getenv("BENCHMARK_ITERATIONS", "3"))
except ValueError:
    logging.warning("Invalid BENCHMARK_ITERATIONS value, using default of 3")
    BENCHMARK_ITERATIONS = 3

# Pause between model benchmarks (in seconds)
try:
    BENCHMARK_PAUSE_SECONDS = float(
        os.getenv("BENCHMARK_PAUSE_SECONDS", "10.0"))
except ValueError:
    logging.warning(
        "Invalid BENCHMARK_PAUSE_SECONDS value, using default of 10.0")
    BENCHMARK_PAUSE_SECONDS = 10.0

# Pause between iterations within a benchmark (in seconds)
try:
    ITERATION_PAUSE_SECONDS = float(
        os.getenv("ITERATION_PAUSE_SECONDS", "1.0"))
except ValueError:
    logging.warning(
        "Invalid ITERATION_PAUSE_SECONDS value, using default of 1.0")
    ITERATION_PAUSE_SECONDS = 1.0

# Maximum token limit for benchmarks
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# =========================================
# Application Settings
# =========================================
# Port for the Gradio app to listen on
APP_PORT = int(os.getenv("APP_PORT", "7860"))

# Whether to allow the app to be accessed from external IP addresses
SHARE_APP = os.getenv("SHARE_APP", "").lower() in ("true", "1", "yes")
