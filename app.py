"""
Gradio web interface for GPT-4o model benchmarking.

This module provides a user interface for running benchmarks on different 
GPT-4o model variants and visualizing the results.
"""

import os
import asyncio
import logging
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import gradio as gr
from typing import List, Dict, Any, Tuple, Optional

from benchmark_runner import BenchmarkRunner
from config import DEFAULT_PROMPT, BENCHMARK_ITERATIONS, BENCHMARK_PAUSE_SECONDS, ITERATION_PAUSE_SECONDS, APP_PORT, SHARE_APP
from utils.exceptions import BenchmarkError

# Configure logger
logger = logging.getLogger(__name__)

# Initialize benchmark runner
benchmark_runner = BenchmarkRunner()


def create_summary_chart(df: pd.DataFrame, metric: str, title: str) -> Optional[go.Figure]:
    """
    Create a bar chart for a specific metric.

    Args:
        df: DataFrame containing the data
        metric: Column name for the metric to visualize
        title: Chart title

    Returns:
        Plotly figure or None if DataFrame is empty
    """
    if df.empty or metric not in df.columns:
        logger.warning(
            f"Cannot create chart for '{metric}': No data or column missing")
        return None

    try:
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
    except Exception as e:
        logger.error(
            f"Error creating chart for '{metric}': {e}", exc_info=True)
        return None


def create_stacked_latency_chart(metrics_dict: Dict[str, List[Dict[str, Any]]]) -> Optional[go.Figure]:
    """
    Create a stacked bar chart showing components of response time.

    Args:
        metrics_dict: Dictionary of metrics by model

    Returns:
        Plotly figure or None if metrics dictionary is empty
    """
    if not metrics_dict:
        return None

    try:
        # Prepare data for the stacked chart
        models = []
        transcription_times = []
        text_times = []
        tts_times = []

        for model_key, metrics_list in metrics_dict.items():
            if not metrics_list:
                continue

            # Calculate averages across iterations
            avg_metrics = {}
            for key in ['model', 'connection_establishment_time', 'text_generation_time', 'tts_time', 'time_to_audio_start', 'whisper_time', 'transcribe_time']:
                if key == 'model':
                    avg_metrics[key] = metrics_list[0][key]
                else:
                    # Not all models have all metrics, use get with default 0
                    avg_metrics[key] = sum(m.get(key, 0)
                                           for m in metrics_list) / len(metrics_list)

            models.append(avg_metrics['model'])

            # Get transcription time - use the appropriate metric name based on the model
            # Some models use whisper_time, others use transcribe_time
            transcription_time = avg_metrics.get(
                'transcribe_time', avg_metrics.get('whisper_time', 0))
            transcription_times.append(transcription_time)

            # For text generation time
            if 'Realtime' in avg_metrics['model'] or 'Voice Live' in avg_metrics['model']:
                # For Realtime/VoiceLive models, use time_to_audio_start minus connection time for text generation
                text_time = avg_metrics.get(
                    'time_to_audio_start', 0) - avg_metrics.get('connection_establishment_time', 0)
                text_times.append(max(0, text_time))  # Ensure non-negative
                tts_times.append(0)  # No separate TTS
            elif 'Audio Preview' in avg_metrics['model']:
                # For Audio model, use text_generation_time (which is the full processing time)
                text_times.append(avg_metrics.get('text_generation_time', 0))
                tts_times.append(0)  # No separate TTS time
            else:
                # For models with TTS (Sequential processing models)
                text_times.append(avg_metrics.get('text_generation_time', 0))
                tts_times.append(avg_metrics.get('tts_time', 0))

        # Create the stacked bar chart
        fig = go.Figure()

        # Add transcription time bars
        fig.add_trace(go.Bar(
            x=models,
            y=transcription_times,
            name='Transcription Time',
            marker_color='#ff9800',
            text=[f"{t:.2f}s" if t > 0 else "" for t in transcription_times],
            textposition='auto'
        ))

        # Add text generation time bars
        fig.add_trace(go.Bar(
            x=models,
            y=text_times,
            name='Text Generation Time',
            marker_color='#1f77b4',
            text=[f"{t:.2f}s" for t in text_times],
            textposition='auto'
        ))

        # Add TTS time bars
        fig.add_trace(go.Bar(
            x=models,
            y=tts_times,
            name='Speech Synthesis Time',
            marker_color='#ff7f0e',
            text=[f"{t:.2f}s" if t > 0 else "" for t in tts_times],
            textposition='auto'
        ))

        # Update layout
        fig.update_layout(
            title='Audio-to-Audio Processing Pipeline Time (seconds)',
            xaxis_title='Model',
            yaxis_title='Time (seconds)',
            barmode='stack',
            legend_title='Component'
        )

        return fig
    except Exception as e:
        logger.error(
            f"Error creating stacked latency chart: {e}", exc_info=True)
        return None


def create_detailed_chart(metrics_dict: Dict[str, List[Dict[str, Any]]], metric: str, title: str) -> Optional[go.Figure]:
    """
    Create a detailed chart showing all iterations for a specific metric.

    Args:
        metrics_dict: Dictionary of metrics by model
        metric: Name of the metric to plot
        title: Chart title

    Returns:
        Plotly figure or None if metrics dictionary is empty
    """
    if not metrics_dict:
        return None

    try:
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
    except Exception as e:
        logger.error(
            f"Error creating detailed chart for '{metric}': {e}", exc_info=True)
        return None


def save_audio_to_file(audio_data: bytes, model_key: str) -> Optional[str]:
    """
    Save audio data to a temporary file with appropriate extension.

    Args:
        audio_data: Binary audio data
        model_key: Key identifying the model that generated the audio

    Returns:
        Path to the saved audio file or None if saving failed
    """
    if not audio_data:
        return None

    try:
        # Determine appropriate extension for the model
        extension = ".wav"
        if model_key == "gpt4o_whisper":
            extension = ".mp3"

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as f:
            f.write(audio_data)
            return f.name
    except Exception as e:
        logger.error(
            f"Error saving audio file for {model_key}: {e}", exc_info=True)
        return None


def create_enhanced_summary_table(df: pd.DataFrame) -> str:
    """
    Create an enhanced HTML summary table with better formatting and relative comparisons.

    Args:
        df: DataFrame containing benchmark results

    Returns:
        HTML formatted table with enhanced styling
    """
    if df.empty:
        return "<p>No data available</p>"

    try:
        # Make a copy to avoid modifying the original dataframe
        formatted_df = df.copy()

        # Define the columns to include in the table and their display names
        display_columns = {
            'model': 'Model',
            'time_to_audio_start': 'Time to Audio Start (s)',
            'audio_duration': 'Audio Duration (s)',
        }

        # Include model-specific columns if they exist
        optional_columns = {
            'tts_time': 'TTS Processing (s)',
            'transcribe_time': 'Transcription Time (s)',
            'audio_size_bytes': 'Audio Size (KB)',
        }

        # Add optional columns that exist in the dataframe
        for col, display in optional_columns.items():
            if col in formatted_df.columns:
                display_columns[col] = display

        # Keep only columns we want to display
        formatted_df = formatted_df[[
            col for col in display_columns.keys() if col in formatted_df.columns]]

        # Calculate relative performance for numeric columns (except 'model')
        numeric_cols = formatted_df.select_dtypes(
            include=['float64', 'int64']).columns

        # Replace NaN values with "N/A" placeholder
        for col in formatted_df.columns:
            if col != 'model':  # Skip the model name column
                formatted_df[col] = formatted_df[col].fillna("N/A")

                # Convert byte sizes to KB for better readability
                if col == 'audio_size_bytes':
                    formatted_df[col] = formatted_df[col].apply(
                        lambda x: round(
                            x / 1024, 1) if isinstance(x, (int, float)) else x
                    )

        # Create HTML for the table with dark mode styling that highlights percentages
        html = """
        <style>
        /* Container for our custom table to isolate styles */
        .benchmark-wrapper {
            background-color: #222 !important;
            padding: 15px !important;
            border-radius: 8px !important;
            margin-bottom: 20px !important;
        }
        
        .benchmark-table {
            border-collapse: collapse !important;
            width: 100% !important;
            font-family: Arial, sans-serif !important;
            margin-bottom: 20px !important;
        }
        
        .benchmark-table th {
            background-color: #2a6334 !important;
            color: #f0f0f0 !important;
            font-weight: bold !important;
            text-align: left !important;
            padding: 12px !important;
            border: 1px solid #444 !important;
        }
        
        .benchmark-table td {
            border: 1px solid #444 !important;
            padding: 10px !important;
            text-align: right !important;
            background-color: #333 !important;
        }
        
        .benchmark-table tr:nth-child(even) td {
            background-color: #2d2d2d !important;
        }
        
        .benchmark-table tr:hover td {
            background-color: #3a3a3a !important;
        }
        
        .model-name {
            font-weight: bold !important;
            text-align: left !important;
        }
                
        /* Percentages and performance indicators */
        .percent-best {
            color: #4caf50 !important;
            font-weight: bold !important;
            font-size: 0.8em !important;
        }
        
        .percent-worst {
            color: #f44336 !important;
            font-size: 0.8em !important;
        }
        
        .percent-mid {
            color: #ffc107 !important;
            font-size: 0.8em !important;
        }
        
        .na-value {
            color: #888 !important;
            font-style: italic !important;
        }
        
        .benchmark-legend {
            background-color: #333 !important;
            padding: 10px !important;
            border-radius: 5px !important;
            margin-top: 15px !important;
            font-size: 0.9em !important;
        }
        
        .legend-item {
            margin: 5px 0 !important;
        }
        
        .legend-best {
            color: #4caf50 !important;
            font-weight: bold !important;
        }
        
        .legend-worst {
            color: #f44336 !important;
        }
        
        .legend-mid {
            color: #ffc107 !important;
        }
        
        .legend-na {
            color: #888 !important;
            font-style: italic !important;
        }
        </style>
        <div class="benchmark-wrapper">
        <table class="benchmark-table">
            <thead>
                <tr>
        """

        # Add table headers
        for col in formatted_df.columns:
            display_name = display_columns.get(
                col, col.replace('_', ' ').title())
            html += f'<th>{display_name}</th>'

        html += '</tr></thead><tbody>'

        # Calculate best and worst values for each numeric column
        best_values = {}
        worst_values = {}

        for col in numeric_cols:
            if col in formatted_df.columns and col != 'model':
                # Get valid numeric values (not 'N/A')
                valid_values = formatted_df[col][formatted_df[col] != 'N/A']
                if not valid_values.empty:
                    # For most metrics, lower is better (except tokens_per_second)
                    if col in ['tokens_per_second', 'audio_duration']:
                        best_values[col] = valid_values.max()
                        worst_values[col] = valid_values.min()
                    else:
                        best_values[col] = valid_values.min()
                        worst_values[col] = valid_values.max()

        # Add table rows
        for _, row in formatted_df.iterrows():
            html += '<tr>'

            for col in formatted_df.columns:
                value = row[col]

                # Format model name cell
                if col == 'model':
                    html += f'<td class="model-name">{value}</td>'
                    continue

                # Format numeric cells with relative performance indicators
                if col in numeric_cols and value != 'N/A':
                    # Format number with appropriate precision
                    if col == 'tokens_per_second':
                        formatted_value = f"{value:.1f}"
                    elif col == 'audio_size_bytes':
                        formatted_value = f"{value:.1f}"
                    else:
                        formatted_value = f"{value:.3f}" if isinstance(
                            value, float) else f"{value}"

                    # Only include percentages for specific columns
                    show_percentage = col in [
                        'time_to_audio_start', 'audio_duration', 'audio_size_bytes']

                    # Add percentage relative to best value
                    if show_percentage and col in best_values and best_values[col] != 0:
                        if col in ['tokens_per_second', 'audio_duration']:
                            # For these metrics, higher is better
                            percent = (value / best_values[col]) * 100
                        else:
                            # For other metrics, lower is better
                            percent = (best_values[col] / value) * 100

                        # Determine percentage class based on performance
                        if value == best_values[col]:
                            percent_class = "percent-best"
                        elif value == worst_values[col]:
                            percent_class = "percent-worst"
                        else:
                            # Middle performance gets a different color
                            percent_class = "percent-mid"

                        html += f'<td>{formatted_value} <span class="{percent_class}">({percent:.0f}%)</span></td>'
                    else:
                        html += f'<td>{formatted_value}</td>'
                else:
                    # Handle non-numeric or N/A values
                    if value == 'N/A':
                        html += '<td class="na-value">Not applicable</td>'
                    else:
                        html += f'<td>{value}</td>'

            html += '</tr>'

        html += '</tbody></table>'

        # Update the legend to reflect the changes in percentage display
        html += """
        <div class="benchmark-legend">
            <p><strong>Legend:</strong></p>
            <ul style="list-style-type: none; padding-left: 10px;">
                <li class="legend-item"><span class="legend-best">(100%)</span>: Best performance for Time to Audio Start, Audio Size, and Audio Duration</li>
                <li class="legend-item"><span class="legend-worst">(0-50%)</span>: Poor relative performance</li>
                <li class="legend-item"><span class="legend-mid">(51-99%)</span>: Intermediate performance</li>
                <li class="legend-item"><span class="legend-na">Not applicable</span>: Metric not available for this model</li>
            </ul>
        </div>
        </div>
        """

        return html

    except Exception as e:
        logger.error(
            f"Error creating enhanced summary table: {e}", exc_info=True)
        # Fallback to standard table rendering if our custom formatting fails
        return df.to_html(float_format="%.3f", index=False, classes="styled-table", na_rep="Not applicable")


async def run_benchmark(
    prompt: str,
    iterations: int,
    include_gpt4o_realtime: bool,
    include_gpt4o_realtime_mini: bool,
    include_gpt4o_audio: bool,
    include_voice_live: bool,
    include_gpt4o_whisper: bool,
    include_azure_speech: bool,
    include_gpt41_mini_whisper: bool,
    include_gpt4o_minitts_transcribe: bool,
    include_gpt4o_minitts_mini_transcribe: bool,
    pause_seconds: float = BENCHMARK_PAUSE_SECONDS,
    iteration_pause: float = ITERATION_PAUSE_SECONDS,
    progress=gr.Progress()
) -> Tuple:
    """
    Run the benchmark and return results for the Gradio interface.

    Args:
        prompt: Text prompt to use for benchmarking
        iterations: Number of iterations to run for each model
        include_gpt4o_realtime: Whether to include the GPT-4o Realtime model
        include_gpt4o_realtime_mini: Whether to include the GPT-4o Realtime Mini model
        include_gpt4o_audio: Whether to include the GPT-4o Audio model
        include_voice_live: Whether to include the Azure Voice Live model
        include_gpt4o_whisper: Whether to include the GPT-4o + Whisper model
        include_azure_speech: Whether to include the Azure Speech model
        include_gpt41_mini_whisper: Whether to include the GPT-4.1-mini + Whisper model
        include_gpt4o_minitts_transcribe: Whether to include GPT-4o + GPT-4o-transcribe + GPT-4o-mini-TTS
        include_gpt4o_minitts_mini_transcribe: Whether to include GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-TTS
        pause_seconds: Number of seconds to pause between model benchmarks
        iteration_pause: Number of seconds to pause between iterations
        progress: Gradio progress indicator

    Returns:
        Tuple containing benchmark results and visualizations
    """
    # Update the pause seconds in config for this run
    global BENCHMARK_PAUSE_SECONDS, ITERATION_PAUSE_SECONDS
    original_pause = BENCHMARK_PAUSE_SECONDS
    original_iteration_pause = ITERATION_PAUSE_SECONDS
    BENCHMARK_PAUSE_SECONDS = pause_seconds
    ITERATION_PAUSE_SECONDS = iteration_pause

    # Determine which models to benchmark
    selected_models = []
    if include_gpt4o_realtime:
        selected_models.append("gpt4o_realtime")
    if include_gpt4o_realtime_mini:
        selected_models.append("gpt4o_realtime_mini")
    if include_gpt4o_audio:
        selected_models.append("gpt4o_audio")
    if include_voice_live:
        selected_models.append("voice_live")
    if include_gpt4o_whisper:
        selected_models.append("gpt4o_whisper")
    if include_azure_speech:
        selected_models.append("azure_speech")
    if include_gpt41_mini_whisper:
        selected_models.append("gpt41_mini_whisper")
    if include_gpt4o_minitts_transcribe:
        selected_models.append("gpt4o_minitts_transcribe")
    if include_gpt4o_minitts_mini_transcribe:
        selected_models.append("gpt4o_minitts_mini_transcribe")

    if not selected_models:
        return ("Please select at least one model.",) + (None,) * 13

    logger.info(
        f"Starting audio-to-audio benchmark with {len(selected_models)} models, {iterations} iterations, {pause_seconds}s pause between models")
    progress(0, desc="Starting benchmark...")

    try:
        # Run the benchmark
        summary_df, detailed_metrics, audio_samples = await benchmark_runner.run_benchmark(
            prompt=prompt,
            iterations=iterations,
            selected_models=selected_models,
            iteration_pause=iteration_pause
        )

        progress(0.8, desc="Generating charts...")

        # Create visualization charts
        stacked_latency_chart = create_stacked_latency_chart(detailed_metrics)
        audio_length_chart = create_summary_chart(
            summary_df, 'audio_duration', 'Generated Audio Duration (seconds)'
        )
        detailed_latency_chart = create_detailed_chart(
            detailed_metrics, 'time_to_audio_start', 'Time To Audio Start by Iteration (seconds)'
        )
        detailed_duration_chart = create_detailed_chart(
            detailed_metrics, 'audio_duration', 'Audio Duration by Iteration (seconds)'
        )

        progress(0.9, desc="Processing audio samples...")

        # Save audio samples to temporary files
        audio_outputs = {}
        for model_key, audio_data in audio_samples.items():
            audio_path = save_audio_to_file(audio_data, model_key)
            if audio_path:
                audio_outputs[model_key] = audio_path

        # Create enhanced HTML summary table
        summary_table = create_enhanced_summary_table(summary_df)

        progress(1.0, desc="Completed!")

        return (
            summary_table,
            stacked_latency_chart,
            audio_length_chart,
            detailed_latency_chart,
            detailed_duration_chart,
            audio_outputs.get("gpt4o_audio", None),
            audio_outputs.get("gpt4o_realtime", None),
            audio_outputs.get("gpt4o_realtime_mini", None),
            audio_outputs.get("voice_live", None),
            audio_outputs.get("gpt4o_whisper", None),
            audio_outputs.get("azure_speech", None),
            audio_outputs.get("gpt41_mini_whisper", None),
            audio_outputs.get("gpt4o_minitts_transcribe", None),
            audio_outputs.get("gpt4o_minitts_mini_transcribe", None)
        )

    except BenchmarkError as e:
        logger.error(f"Benchmark error: {e}", exc_info=True)
        return (f"Benchmark error: {str(e)}",) + (None,) * 13
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return (f"An unexpected error occurred: {str(e)}",) + (None,) * 13
    finally:
        # Restore original settings
        BENCHMARK_PAUSE_SECONDS = original_pause
        ITERATION_PAUSE_SECONDS = original_iteration_pause


def create_gradio_interface() -> gr.Blocks:
    """
    Create the Gradio interface for the benchmarking application.

    Returns:
        Configured Gradio Blocks interface
    """
    with gr.Blocks(title="GPT-4o Benchmarking Suite", theme=gr.themes.Base()) as app:
        gr.Markdown("# GPT-4o Model Benchmarking Suite")

        with gr.Row():
            with gr.Column():
                prompt_input = gr.Textbox(
                    label="Benchmark Prompt",
                    placeholder="Enter text prompt here...",
                    value=DEFAULT_PROMPT,
                    lines=5
                )

                with gr.Row():
                    iterations_input = gr.Slider(
                        label="Benchmark Iterations",
                        minimum=1,
                        maximum=10,
                        value=BENCHMARK_ITERATIONS,
                        step=1
                    )

                    pause_input = gr.Slider(
                        label="Pause Between Models (seconds)",
                        minimum=0,
                        maximum=30,
                        value=BENCHMARK_PAUSE_SECONDS,
                        step=1
                    )

                    iteration_pause_input = gr.Slider(
                        label="Pause Between Iterations (seconds)",
                        minimum=0,
                        maximum=10,
                        value=ITERATION_PAUSE_SECONDS,
                        step=0.5
                    )

                # Model selection with better organization
                gr.Markdown("### Select Models to Benchmark")

                with gr.Group():
                    gr.Markdown("#### 🚀 End-to-End Options")
                    with gr.Row():
                        include_gpt4o_realtime = gr.Checkbox(
                            label="GPT-4o Realtime", value=True)
                        include_gpt4o_realtime_mini = gr.Checkbox(
                            label="GPT-4o Realtime Mini", value=True)
                        include_gpt4o_audio = gr.Checkbox(
                            label="GPT-4o Audio", value=True)
                        include_voice_live = gr.Checkbox(
                            label="Azure Voice Live (Experimental)", value=False)

                with gr.Group():
                    gr.Markdown("#### 🔗 Cascading Pipeline Models")
                    with gr.Row():
                        include_gpt4o_whisper = gr.Checkbox(
                            label="GPT-4o + Whisper + TTS", value=True)
                        include_gpt4o_minitts_transcribe = gr.Checkbox(
                            label="GPT-4o + GPT-4o-transcribe + GPT-4o-mini-tts", value=True)

                    with gr.Row():
                        include_gpt4o_minitts_mini_transcribe = gr.Checkbox(
                            label="GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-tts", value=True)
                        include_gpt41_mini_whisper = gr.Checkbox(
                            label="GPT-4.1-mini + Whisper + TTS", value=True)

                with gr.Group():
                    gr.Markdown("#### ⚡ Production-Ready Solutions")
                    with gr.Row():
                        include_azure_speech = gr.Checkbox(
                            label="Azure Speech + OpenAI", value=True)

                run_button = gr.Button(
                    "Run Benchmark", variant="primary", size="lg")

        with gr.Tabs():
            with gr.TabItem("Summary Charts"):
                gr.Markdown("### Benchmark Summary")
                summary_table = gr.HTML()

                with gr.Row():
                    stacked_latency_chart = gr.Plot(
                        label="Time Until Audio Playback Start")
                    audio_length_chart = gr.Plot(
                        label="Generated Audio Duration")

                # Remove tokens_per_second row and just keep a placeholder for layout balance
                gr.HTML("<div style='height: 20px;'></div>")

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

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### GPT-4o Realtime Mini Sample")
                        gpt4o_realtime_mini_output = gr.Audio(
                            label="GPT-4o Realtime Mini", type="filepath")

                    with gr.Column():
                        gr.Markdown("#### Azure Voice Live Sample")
                        voice_live_output = gr.Audio(
                            label="Azure Voice Live", type="filepath")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### GPT-4o + Whisper + TTS Sample")
                        gpt4o_whisper_output = gr.Audio(
                            label="GPT-4o + Whisper + TTS", type="filepath")

                    with gr.Column():
                        gr.Markdown("#### GPT-4.1-mini + Whisper + TTS Sample")
                        gpt41_mini_whisper_output = gr.Audio(
                            label="GPT-4.1-mini + Whisper + TTS", type="filepath")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Azure Speech Sample")
                        azure_speech_output = gr.Audio(
                            label="Azure Speech", type="filepath")

                    with gr.Column():
                        gr.Markdown(
                            "#### GPT-4o + GPT-4o-transcribe + GPT-4o-mini-tts Sample")
                        gpt4o_minitts_transcribe_output = gr.Audio(
                            label="GPT-4o + GPT-4o-transcribe + GPT-4o-mini-tts", type="filepath")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "#### GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-tts Sample")
                        gpt4o_minitts_mini_transcribe_output = gr.Audio(
                            label="GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-tts", type="filepath")

        run_button.click(
            fn=run_benchmark,
            inputs=[
                prompt_input,
                iterations_input,
                include_gpt4o_realtime,
                include_gpt4o_realtime_mini,
                include_gpt4o_audio,
                include_voice_live,
                include_gpt4o_whisper,
                include_azure_speech,
                include_gpt41_mini_whisper,
                include_gpt4o_minitts_transcribe,
                include_gpt4o_minitts_mini_transcribe,
                pause_input,
                iteration_pause_input
            ],
            outputs=[
                summary_table,
                stacked_latency_chart,
                audio_length_chart,
                detailed_latency_chart,
                detailed_duration_chart,
                gpt4o_audio_output,
                gpt4o_realtime_output,
                gpt4o_realtime_mini_output,
                voice_live_output,
                gpt4o_whisper_output,
                gpt41_mini_whisper_output,
                azure_speech_output,
                gpt4o_minitts_transcribe_output,
                gpt4o_minitts_mini_transcribe_output
            ]
        )

    return app


if __name__ == "__main__":
    try:
        # Create and launch the Gradio interface
        app = create_gradio_interface()
        app.launch(
            server_port=APP_PORT,
            share=SHARE_APP,
            server_name="0.0.0.0"  # Listen on all interfaces
        )
    except Exception as e:
        logger.error(f"Error launching application: {e}", exc_info=True)
