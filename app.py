import os
import asyncio
import base64
import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any

from benchmark_runner import BenchmarkRunner
from config import DEFAULT_PROMPT, BENCHMARK_ITERATIONS

# Initialize benchmark runner
benchmark_runner = BenchmarkRunner()


def create_summary_chart(df: pd.DataFrame, metric: str, title: str):
    """Create a bar chart for a specific metric."""
    if df.empty:
        return None

    fig = px.bar(
        df,
        x='model',
        y=metric,
        title=title,
        color='model',
        text_auto='.3f'
    )
    fig.update_layout(xaxis_title="Model", yaxis_title=metric)
    return fig


def create_stacked_latency_chart(metrics_dict: Dict[str, List[Dict[str, Any]]]):
    """Create a stacked bar chart showing components of response time."""
    if not metrics_dict:
        return None

    # Prepare data for the stacked chart
    models = []
    text_times = []
    tts_times = []

    for model_key, metrics_list in metrics_dict.items():
        if not metrics_list:
            continue

        # Calculate averages across iterations
        avg_metrics = {}
        for key in ['model', 'text_generation_time', 'tts_time', 'time_to_audio_start']:
            if key == 'model':
                avg_metrics[key] = metrics_list[0][key]
            else:
                # Not all models have all metrics, use get with default 0
                avg_metrics[key] = sum(m.get(key, 0)
                                       for m in metrics_list) / len(metrics_list)

        models.append(avg_metrics['model'])

        # For Whisper model, we have both text and TTS components
        if 'Whisper' in avg_metrics['model']:
            text_times.append(avg_metrics.get('text_generation_time', 0))
            tts_times.append(avg_metrics.get('tts_time', 0))
        elif 'Realtime' in avg_metrics['model']:
            # For Realtime model, use time_to_audio_start directly
            text_times.append(avg_metrics.get('time_to_audio_start', 0))
            tts_times.append(0)  # No separate TTS
        else:
            # For Audio model, use generation time
            text_times.append(avg_metrics.get('text_generation_time', 0))
            tts_times.append(0)  # No separate TTS time for other models

    # Create the stacked bar chart
    fig = go.Figure()

    # Add text generation time bars
    fig.add_trace(go.Bar(
        x=models,
        y=text_times,
        name='Text Generation Time',
        marker_color='#1f77b4',
        text=[f"{t:.2f}s" for t in text_times],
        textposition='auto'
    ))

    # Add TTS time bars (will be zero for models without TTS)
    fig.add_trace(go.Bar(
        x=models,
        y=tts_times,
        name='TTS Processing Time',
        marker_color='#ff7f0e',
        text=[f"{t:.2f}s" if t > 0 else "" for t in tts_times],
        textposition='auto'
    ))

    # Update layout
    fig.update_layout(
        title='Time Until Audio Playback Start (seconds)',
        xaxis_title='Model',
        yaxis_title='Time (seconds)',
        barmode='stack',
        legend_title='Component'
    )

    return fig


def create_detailed_chart(metrics_dict: Dict[str, List[Dict[str, Any]]], metric: str, title: str):
    """Create a detailed chart showing all iterations for a specific metric."""
    if not metrics_dict:
        return None

    fig = go.Figure()

    for model_key, metrics_list in metrics_dict.items():
        if not metrics_list:
            continue

        model_name = metrics_list[0]["model"]
        metric_values = [m.get(metric, 0) for m in metrics_list]
        iterations = list(range(1, len(metrics_list) + 1))

        fig.add_trace(go.Scatter(
            x=iterations,
            y=metric_values,
            mode='lines+markers',
            name=model_name
        ))

    fig.update_layout(
        title=title,
        xaxis_title="Iteration",
        yaxis_title=metric,
        legend_title="Model"
    )
    return fig


async def run_benchmark(prompt: str,
                        iterations: int,
                        include_gpt4o_realtime: bool,
                        include_gpt4o_audio: bool,
                        include_gpt4o_whisper: bool,
                        progress=gr.Progress()):
    """Run the benchmark and return results for the Gradio interface."""

    selected_models = []
    if include_gpt4o_realtime:
        selected_models.append("gpt4o_realtime")
    if include_gpt4o_audio:
        selected_models.append("gpt4o_audio")
    if include_gpt4o_whisper:
        selected_models.append("gpt4o_whisper")

    if not selected_models:
        return ("Please select at least one model.", None, None, None, None, None, None, None, None, None)

    progress(0, desc="Starting benchmark...")

    summary_df, detailed_metrics, audio_samples = await benchmark_runner.run_benchmark(
        prompt=prompt,
        iterations=iterations,
        selected_models=selected_models
    )

    progress(1, desc="Generating charts...")

    # Create stacked latency chart showing components (use this for time_to_audio_start)
    stacked_latency_chart = create_stacked_latency_chart(detailed_metrics)

    # Use audio_duration for the actual audio length
    audio_length_chart = create_summary_chart(
        summary_df, 'audio_duration', 'Generated Audio Duration (seconds)'
    )

    tokens_per_second_chart = create_summary_chart(
        summary_df, 'tokens_per_second', 'Tokens Per Second'
    )

    # Create detailed charts
    detailed_latency_chart = create_detailed_chart(
        detailed_metrics, 'time_to_audio_start', 'Time To Audio Start by Iteration (seconds)'
    )

    detailed_duration_chart = create_detailed_chart(
        detailed_metrics, 'audio_duration', 'Audio Duration by Iteration (seconds)'
    )

    # Prepare audio samples
    audio_outputs = {}
    for model_key, audio_data in audio_samples.items():
        if audio_data:
            # Create a temporary file for the audio with appropriate extension
            # Use .wav for PCM/WAV data (realtime model) and .mp3 for whisper model
            extension = ".wav"
            if model_key == "gpt4o_whisper":
                extension = ".mp3"

            temp_filename = f"temp_{model_key}{extension}"
            with open(temp_filename, "wb") as f:
                f.write(audio_data)
            audio_outputs[model_key] = temp_filename

    # Create data table
    summary_table = summary_df.to_html(
        float_format="%.3f",
        index=False,
        classes="styled-table"
    )

    return (
        summary_table,
        stacked_latency_chart,  # Replace latency_chart with stacked_latency_chart
        audio_length_chart,
        tokens_per_second_chart,
        detailed_latency_chart,
        detailed_duration_chart,
        audio_outputs.get("gpt4o_audio", None),
        audio_outputs.get("gpt4o_realtime", None),
        audio_outputs.get("gpt4o_whisper", None)
    )

# Create Gradio interface
with gr.Blocks(title="GPT-4o Benchmarking Suite", theme=gr.themes.Base()) as app:
    gr.Markdown("# GPT-4o Model Benchmarking Suite")
    gr.Markdown("""
    Compare performance metrics across different GPT-4o model variants:
    - GPT-4o Realtime 
    - GPT-4o Audio Preview
    - GPT-4o + Whisper TTS
    """)

    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Benchmark Prompt",
                placeholder="Enter text prompt here...",
                value=DEFAULT_PROMPT,
                lines=5
            )

            iterations_input = gr.Slider(
                label="Benchmark Iterations",
                minimum=1,
                maximum=10,
                value=BENCHMARK_ITERATIONS,
                step=1
            )

            with gr.Row():
                include_gpt4o_realtime = gr.Checkbox(
                    label="GPT-4o Realtime", value=True)
                include_gpt4o_audio = gr.Checkbox(
                    label="GPT-4o Audio", value=True)
                include_gpt4o_whisper = gr.Checkbox(
                    label="GPT-4o + Whisper", value=True)

            run_button = gr.Button("Run Benchmark", variant="primary")

    with gr.Tabs():
        with gr.TabItem("Summary Charts"):
            gr.Markdown("### Benchmark Summary")
            summary_table = gr.HTML()

            with gr.Row():
                stacked_latency_chart = gr.Plot(
                    label="Time Until Audio Playback Start")
                audio_length_chart = gr.Plot(label="Generated Audio Duration")

            with gr.Row():
                tokens_per_second_chart = gr.Plot(label="Tokens Per Second")
                # Empty space for layout balance
                gr.HTML("<div style='height: 400px;'></div>")

        with gr.TabItem("Detailed Charts"):
            gr.Markdown("### Detailed Performance by Iteration")

            with gr.Row():
                detailed_latency_chart = gr.Plot(
                    label="Time To Audio Start by Iteration")
                detailed_duration_chart = gr.Plot(
                    label="Audio Duration by Iteration")

        with gr.TabItem("Audio Samples"):
            gr.Markdown("### Audio Output Samples")

            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### GPT-4o Audio Sample")
                    gpt4o_audio_output = gr.Audio(
                        label="GPT-4o Audio", type="filepath")

                with gr.Column():
                    gr.Markdown("#### GPT-4o Realtime Sample")
                    gpt4o_realtime_output = gr.Audio(
                        label="GPT-4o Realtime", type="filepath")

                with gr.Column():
                    gr.Markdown("#### GPT-4o + Whisper Sample")
                    gpt4o_whisper_output = gr.Audio(
                        label="GPT-4o + Whisper", type="filepath")

    run_button.click(
        fn=run_benchmark,
        inputs=[
            prompt_input,
            iterations_input,
            include_gpt4o_realtime,
            include_gpt4o_audio,
            include_gpt4o_whisper
        ],
        outputs=[
            summary_table,
            stacked_latency_chart,  # Remove redundant latency_chart
            audio_length_chart,
            tokens_per_second_chart,
            detailed_latency_chart,
            detailed_duration_chart,
            gpt4o_audio_output,
            gpt4o_realtime_output,
            gpt4o_whisper_output
        ]
    )

if __name__ == "__main__":
    app.launch()
