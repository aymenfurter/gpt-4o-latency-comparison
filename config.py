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
# Model Deployment Names
# =========================================
GPT4O_DEPLOYMENT = os.getenv("GPT4O_DEPLOYMENT", "gpt-4o")
GPT4O_REALTIME_DEPLOYMENT = os.getenv(
    "GPT4O_REALTIME_DEPLOYMENT", "gpt-4o-realtime-preview"
)
GPT4O_AUDIO_DEPLOYMENT = os.getenv(
    "GPT4O_AUDIO_DEPLOYMENT", "gpt-4o-audio-preview"
)
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

# Maximum token limit for benchmarks
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# =========================================
# Application Settings
# =========================================
# Port for the Gradio app to listen on
APP_PORT = int(os.getenv("APP_PORT", "7860"))

# Whether to allow the app to be accessed from external IP addresses
SHARE_APP = os.getenv("SHARE_APP", "").lower() in ("true", "1", "yes")
