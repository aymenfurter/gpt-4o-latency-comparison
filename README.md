# GPT-4o Latency Comparison

A benchmarking toolkit for comparing performance metrics between different GPT-4o implementations:
- GPT-4o Realtime
- GPT-4o Audio Preview
- GPT-4o + Whisper TTS

## Features

- Compare response times across different models
- Measure and visualize time to audio playback start
- Track token generation rates
- Analyze audio durations
- Interactive web interface using Gradio

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt-4o-latency-comparison.git
cd gpt-4o-latency-comparison
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install FFmpeg (required for audio duration detection):
```bash
# For Debian/Ubuntu
apt-get update && apt-get install -y ffmpeg

# For macOS
brew install ffmpeg

# For Windows
# Download from https://ffmpeg.org/download.html
```

4. Create a `.env` file with your Azure OpenAI credentials:
```
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_API_VERSION=2025-01-01-preview
GPT4O_DEPLOYMENT=your-gpt4o-deployment-name
GPT4O_REALTIME_DEPLOYMENT=your-gpt4o-realtime-deployment-name
GPT4O_AUDIO_DEPLOYMENT=your-gpt4o-audio-deployment-name
TTS_DEPLOYMENT=your-tts-deployment-name
```

## Usage

1. Start the benchmarking interface:
```bash
python app.py
```

2. Open the URL displayed in the console (typically http://127.0.0.1:7860)

3. Enter a prompt, select the desired models, set the number of iterations, and click "Run Benchmark"

4. Explore the results with interactive charts and audio samples

## Metrics Explained

- **Time Until Audio Playback Start**: How long it takes from sending the request until audio would begin playing
  - For GPT-4o Realtime: Time until the first token is received
  - For GPT-4o Audio: Total processing time until audio is returned
  - For GPT-4o + Whisper TTS: Combined time for text generation and TTS processing

- **Generated Audio Duration**: Actual length of the generated audio in seconds

- **Tokens Per Second**: Text token generation speed

## Dependencies

- openai
- pydub
- ffmpeg (for audio duration analysis)
- gradio
- pandas
- plotly
- azure-identity
- python-dotenv

## Notes

- FFmpeg is required for accurate MP3 duration detection in the GPT-4o + Whisper TTS model.
- The benchmark computes an average across the specified number of iterations for more reliable results.
- When comparing models, consider both speed (time to audio start) and quality (tokens per second).
