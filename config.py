import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Azure OpenAI Configuration
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# Model deployments
GPT4O_DEPLOYMENT = os.getenv("GPT4O_DEPLOYMENT", "gpt-4o")
GPT4O_REALTIME_DEPLOYMENT = os.getenv(
    "GPT4O_REALTIME_DEPLOYMENT", "gpt-4o-realtime-preview")
GPT4O_AUDIO_DEPLOYMENT = os.getenv(
    "GPT4O_AUDIO_DEPLOYMENT", "gpt-4o-audio-preview")
WHISPER_DEPLOYMENT = os.getenv("WHISPER_DEPLOYMENT", "whisper")
TTS_DEPLOYMENT = os.getenv("TTS_DEPLOYMENT", "tts")

# Benchmark settings
# Define a concise test prompt for benchmark comparisons
DEFAULT_PROMPT = "Translate the word 'trainstation' to Spanish. Respond with only a single word."
BENCHMARK_ITERATIONS = int(os.getenv("BENCHMARK_ITERATIONS", "3"))
