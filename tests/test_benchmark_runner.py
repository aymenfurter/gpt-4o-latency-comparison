"""Tests for benchmark runner."""
import pytest
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from benchmark_runner import BenchmarkRunner
from utils.exceptions import BenchmarkError, ModelInitializationError


class MockModel:
    """Mock model for testing."""
    
    def __init__(self, name="MockModel"):
        self.name = name
        self.initialized = False
        self.should_fail_init = False
        self.should_fail_generate = False
        
    async def initialize(self):
        """Mock initialize method."""
        if self.should_fail_init:
            raise ModelInitializationError(f"Failed to initialize {self.name}")
        self.initialized = True
    
    async def generate_response_from_audio(self, audio_data, text_prompt=None):
        """Mock generate_response_from_audio method."""
        if self.should_fail_generate:
            raise Exception(f"Generation failed for {self.name}")
        
        return (
            f"Response from {self.name}",
            {
                "model": self.name,
                "processing_time": 1.5,
                "time_to_audio_start": 1.0,
                "audio_duration": 3.0,
                "tokens_per_second": 10.0,
                "whisper_time": 0.5,
                "audio_size_bytes": 1024
            },
            b"mock_audio_data"
        )


class TestBenchmarkRunner:
    """Test the BenchmarkRunner class."""
    
    def test_init(self):
        """Test BenchmarkRunner initialization."""
        runner = BenchmarkRunner()
        
        # Check that models dictionary is populated
        assert len(runner.models) == 8
        assert "gpt4o_realtime" in runner.models
        assert "gpt4o_realtime_mini" in runner.models
        assert "gpt4o_audio" in runner.models
        assert "gpt4o_whisper" in runner.models
        assert "azure_speech" in runner.models
        assert "gpt41_mini_whisper" in runner.models
        assert "gpt4o_minitts_transcribe" in runner.models
        assert "gpt4o_minitts_mini_transcribe" in runner.models
        
        assert runner.results == {}
        assert runner.audio_input is None
        assert runner.prompt_text is None
    
    def test_get_target_models_all(self):
        """Test _get_target_models with no selection (all models)."""
        runner = BenchmarkRunner()
        
        target_models = runner._get_target_models(None)
        
        assert len(target_models) == 8
        assert target_models == runner.models
    
    def test_get_target_models_selected(self):
        """Test _get_target_models with specific model selection."""
        runner = BenchmarkRunner()
        selected = ["gpt4o_audio", "gpt4o_whisper"]
        
        target_models = runner._get_target_models(selected)
        
        assert len(target_models) == 2
        assert "gpt4o_audio" in target_models
        assert "gpt4o_whisper" in target_models
        assert "gpt4o_realtime" not in target_models
    
    def test_get_target_models_nonexistent(self):
        """Test _get_target_models with nonexistent model selection."""
        runner = BenchmarkRunner()
        selected = ["nonexistent_model", "gpt4o_audio"]
        
        target_models = runner._get_target_models(selected)
        
        assert len(target_models) == 1
        assert "gpt4o_audio" in target_models
        assert "nonexistent_model" not in target_models
    
    @pytest.mark.asyncio
    async def test_initialize_models_all(self):
        """Test initialize_models with all models."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model1 = MockModel("Model1")
        mock_model2 = MockModel("Model2")
        runner.models = {"model1": mock_model1, "model2": mock_model2}
        
        await runner.initialize_models()
        
        assert mock_model1.initialized is True
        assert mock_model2.initialized is True
    
    @pytest.mark.asyncio
    async def test_initialize_models_selected(self):
        """Test initialize_models with selected models."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model1 = MockModel("Model1")
        mock_model2 = MockModel("Model2")
        runner.models = {"model1": mock_model1, "model2": mock_model2}
        
        await runner.initialize_models(["model1"])
        
        assert mock_model1.initialized is True
        assert mock_model2.initialized is False
    
    @pytest.mark.asyncio
    async def test_initialize_models_with_failure(self):
        """Test initialize_models when a model fails to initialize."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model1 = MockModel("Model1")
        mock_model2 = MockModel("Model2")
        mock_model2.should_fail_init = True
        runner.models = {"model1": mock_model1, "model2": mock_model2}
        
        with patch('benchmark_runner.logger') as mock_logger:
            await runner.initialize_models()
            
            # Should log error but continue
            mock_logger.error.assert_called()
        
        assert mock_model1.initialized is True
        assert mock_model2.initialized is False
    
    def test_get_common_metric_keys_empty(self):
        """Test _get_common_metric_keys with empty list."""
        runner = BenchmarkRunner()
        
        result = runner._get_common_metric_keys([])
        
        assert result == set()
    
    def test_get_common_metric_keys_single(self):
        """Test _get_common_metric_keys with single metrics dict."""
        runner = BenchmarkRunner()
        metrics_list = [{"a": 1, "b": 2, "c": 3}]
        
        result = runner._get_common_metric_keys(metrics_list)
        
        assert result == {"a", "b", "c"}
    
    def test_get_common_metric_keys_multiple(self):
        """Test _get_common_metric_keys with multiple metrics dicts."""
        runner = BenchmarkRunner()
        metrics_list = [
            {"a": 1, "b": 2, "c": 3},
            {"a": 4, "b": 5, "d": 6},
            {"a": 7, "e": 8}
        ]
        
        result = runner._get_common_metric_keys(metrics_list)
        
        assert result == {"a"}  # Only "a" is common to all
    
    def test_create_summary_dataframe_empty(self):
        """Test _create_summary_dataframe with empty metrics."""
        runner = BenchmarkRunner()
        
        df = runner._create_summary_dataframe({})
        
        assert df.empty
    
    def test_create_summary_dataframe_normal(self):
        """Test _create_summary_dataframe with normal metrics."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model = MockModel("TestModel")
        runner.models = {"test_model": mock_model}
        
        metrics_dict = {
            "test_model": [
                {
                    "processing_time": 1.0,
                    "time_to_audio_start": 0.8,
                    "audio_duration": 2.0,
                    "tokens_per_second": 5.0,
                    "whisper_time": 0.3,
                    "audio_size_bytes": 1024
                },
                {
                    "processing_time": 2.0,
                    "time_to_audio_start": 1.2,
                    "audio_duration": 3.0,
                    "tokens_per_second": 7.0,
                    "whisper_time": 0.5,
                    "audio_size_bytes": 2048
                }
            ]
        }
        
        df = runner._create_summary_dataframe(metrics_dict)
        
        assert len(df) == 1
        assert df.iloc[0]["model"] == "TestModel"
        assert df.iloc[0]["processing_time"] == 1.5  # (1.0 + 2.0) / 2
        assert df.iloc[0]["time_to_audio_start"] == 1.0  # (0.8 + 1.2) / 2
        assert df.iloc[0]["audio_duration"] == 2.5  # (2.0 + 3.0) / 2
        assert df.iloc[0]["tokens_per_second"] == 6.0  # (5.0 + 7.0) / 2
        assert df.iloc[0]["whisper_time"] == 0.4  # (0.3 + 0.5) / 2
        assert df.iloc[0]["audio_size_bytes"] == 1536  # (1024 + 2048) / 2
    
    def test_create_summary_dataframe_missing_audio_bytes(self):
        """Test _create_summary_dataframe without audio_size_bytes."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model = MockModel("TestModel")
        runner.models = {"test_model": mock_model}
        
        metrics_dict = {
            "test_model": [
                {
                    "processing_time": 1.0,
                    "time_to_audio_start": 0.8,
                    "audio_duration": 2.0,
                    "tokens_per_second": 5.0,
                    "whisper_time": 0.3
                }
            ]
        }
        
        df = runner._create_summary_dataframe(metrics_dict)
        
        assert len(df) == 1
        assert df.iloc[0]["model"] == "TestModel"
        assert "audio_size_bytes" not in df.columns
    
    def test_create_summary_dataframe_with_model_specific_metrics(self):
        """Test _create_summary_dataframe with model-specific metrics."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model = MockModel("TestModel")
        runner.models = {"test_model": mock_model}
        
        metrics_dict = {
            "test_model": [
                {
                    "processing_time": 1.0,
                    "time_to_audio_start": 0.8,
                    "audio_duration": 2.0,
                    "tokens_per_second": 5.0,
                    "whisper_time": 0.3,
                    "custom_metric": 10.0
                },
                {
                    "processing_time": 2.0,
                    "time_to_audio_start": 1.2,
                    "audio_duration": 3.0,
                    "tokens_per_second": 7.0,
                    "whisper_time": 0.5,
                    "custom_metric": 20.0
                }
            ]
        }
        
        df = runner._create_summary_dataframe(metrics_dict)
        
        assert len(df) == 1
        assert df.iloc[0]["custom_metric"] == 15.0  # (10.0 + 20.0) / 2
    
    @pytest.mark.asyncio
    @patch('benchmark_runner.requests.post')
    async def test_generate_audio_input_success(self, mock_post):
        """Test _generate_audio_input with successful TTS."""
        runner = BenchmarkRunner()
        
        # Mock successful TTS response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_post.return_value = mock_response
        
        with patch('benchmark_runner.AZURE_OPENAI_API_KEY', 'test-key'):
            audio_data, prompt_text = await runner._generate_audio_input("Test prompt")
        
        assert audio_data == b"fake_audio_data"
        assert prompt_text == "Test prompt"
        mock_post.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('benchmark_runner.requests.post')
    async def test_generate_audio_input_failure(self, mock_post):
        """Test _generate_audio_input with TTS failure."""
        runner = BenchmarkRunner()
        
        # Mock failed TTS response
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "TTS Error"
        mock_post.return_value = mock_response
        
        with patch('benchmark_runner.AZURE_OPENAI_API_KEY', 'test-key'):
            with pytest.raises(BenchmarkError):
                await runner._generate_audio_input("Test prompt")
    
    @pytest.mark.asyncio
    @patch('azure.identity.aio.DefaultAzureCredential')
    @patch('benchmark_runner.requests.post')
    async def test_generate_audio_input_with_entra_id(self, mock_post, mock_credential):
        """Test _generate_audio_input with Entra ID authentication."""
        runner = BenchmarkRunner()
        
        # Mock credential and token
        mock_token = Mock()
        mock_token.token = "test-token"
        mock_credential_instance = Mock()
        mock_credential_instance.get_token = AsyncMock(return_value=mock_token)
        mock_credential.return_value = mock_credential_instance
        
        # Mock successful TTS response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b"fake_audio_data"
        mock_post.return_value = mock_response
        
        with patch('benchmark_runner.AZURE_OPENAI_API_KEY', None):
            audio_data, prompt_text = await runner._generate_audio_input("Test prompt")
        
        assert audio_data == b"fake_audio_data"
        assert prompt_text == "Test prompt"
    
    @pytest.mark.asyncio
    async def test_run_benchmark_success(self):
        """Test run_benchmark with successful execution."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model = MockModel("TestModel")
        runner.models = {"test_model": mock_model}
        
        with patch.object(runner, '_generate_audio_input') as mock_gen_audio:
            mock_gen_audio.return_value = (b"test_audio", "test prompt")
            
            summary_df, metrics, audio = await runner.run_benchmark(
                prompt="Test prompt",
                iterations=1,
                selected_models=["test_model"],
                iteration_pause=0.1
            )
        
        # Check results
        assert len(summary_df) == 1
        assert "test_model" in metrics
        assert len(metrics["test_model"]) == 1
        assert "test_model" in audio
    
    @pytest.mark.asyncio
    async def test_run_benchmark_with_model_failure(self):
        """Test run_benchmark when a model fails to generate."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model = MockModel("TestModel")
        mock_model.should_fail_generate = True
        runner.models = {"test_model": mock_model}
        
        with patch.object(runner, '_generate_audio_input') as mock_gen_audio:
            mock_gen_audio.return_value = (b"test_audio", "test prompt")
            
            with patch('benchmark_runner.logger') as mock_logger:
                summary_df, metrics, audio = await runner.run_benchmark(
                    prompt="Test prompt",
                    iterations=1,
                    selected_models=["test_model"],
                    iteration_pause=0.1
                )
                
                # Should log error
                mock_logger.error.assert_called()
        
        # Check that we still get results (empty for failed model)
        assert "test_model" in metrics
        assert len(metrics["test_model"]) == 0  # No successful runs
    
    @pytest.mark.asyncio
    async def test_run_benchmark_defaults(self):
        """Test run_benchmark with default parameters."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model = MockModel("TestModel")
        runner.models = {"test_model": mock_model}
        
        with patch.object(runner, '_generate_audio_input') as mock_gen_audio:
            mock_gen_audio.return_value = (b"test_audio", "default prompt")
            
            with patch('benchmark_runner.DEFAULT_PROMPT', 'default prompt'):
                with patch('benchmark_runner.BENCHMARK_ITERATIONS', 1):
                    summary_df, metrics, audio = await runner.run_benchmark()
        
        # Should use defaults and run on all models
        assert len(summary_df) == 1  # We only have one mock model
        mock_gen_audio.assert_called_once_with('default prompt')
    
    @pytest.mark.asyncio
    async def test_run_benchmark_with_pause(self):
        """Test run_benchmark with iteration pause."""
        runner = BenchmarkRunner()
        
        # Replace models with mocks
        mock_model1 = MockModel("TestModel1")
        mock_model2 = MockModel("TestModel2")
        runner.models = {"model1": mock_model1, "model2": mock_model2}
        
        with patch.object(runner, '_generate_audio_input') as mock_gen_audio:
            mock_gen_audio.return_value = (b"test_audio", "test prompt")
            
            with patch('asyncio.sleep') as mock_sleep:
                with patch('benchmark_runner.BENCHMARK_PAUSE_SECONDS', 0.1):
                    summary_df, metrics, audio = await runner.run_benchmark(
                        prompt="Test prompt",
                        iterations=1,
                        iteration_pause=0.1
                    )
                    
                    # Should have called sleep for pauses between models and iterations
                    mock_sleep.assert_called()
        
        assert len(summary_df) == 2  # Two models