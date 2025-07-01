"""Tests for base model and configuration."""
import pytest
import time
import os
from unittest.mock import Mock, patch, AsyncMock
from models.base_model import BaseModel
from utils.exceptions import ModelInitializationError, ModelGenerationError
import config


class TestConcreteModel(BaseModel):
    """Concrete implementation of BaseModel for testing."""
    
    def __init__(self, name="TestModel"):
        super().__init__(name)
        self.init_called = False
        self.should_fail_init = False
    
    async def initialize(self):
        """Test implementation of initialize."""
        if self.should_fail_init:
            raise ModelInitializationError("Test initialization failed")
        self.init_called = True
    
    async def generate_response_from_audio(self, audio_data, text_prompt=None):
        """Test implementation of generate_response_from_audio."""
        return "test response", {"test": "metrics"}, b"test_audio"


class TestBaseModel:
    """Test the BaseModel abstract class."""
    
    def test_init(self):
        """Test BaseModel initialization."""
        model = TestConcreteModel("TestModel")
        assert model.name == "TestModel"
        assert model.client is None
        assert model.initialized is False
    
    @pytest.mark.asyncio
    async def test_ensure_initialized_success(self):
        """Test ensure_initialized with successful initialization."""
        model = TestConcreteModel()
        
        await model.ensure_initialized()
        
        assert model.init_called is True
        assert model.initialized is True
    
    @pytest.mark.asyncio
    async def test_ensure_initialized_already_initialized(self):
        """Test ensure_initialized when already initialized."""
        model = TestConcreteModel()
        model.initialized = True
        
        await model.ensure_initialized()
        
        # Should not call initialize again
        assert model.init_called is False
    
    @pytest.mark.asyncio
    async def test_ensure_initialized_failure(self):
        """Test ensure_initialized with initialization failure."""
        model = TestConcreteModel()
        model.should_fail_init = True
        
        with pytest.raises(ModelInitializationError):
            await model.ensure_initialized()
        
        assert model.initialized is False
    
    def test_collect_basic_metrics_without_first_token(self):
        """Test collect_basic_metrics without first token time."""
        model = TestConcreteModel()
        start_time = time.time() - 5.0  # 5 seconds ago
        
        with patch('time.time', return_value=start_time + 5.0):
            metrics = model.collect_basic_metrics(start_time)
        
        assert metrics["model"] == "TestModel"
        assert metrics["processing_time"] == 5.0
        assert metrics["audio_duration"] == 0.0
        assert metrics["time_to_audio_start"] == 0.0
    
    def test_collect_basic_metrics_with_first_token(self):
        """Test collect_basic_metrics with first token time."""
        model = TestConcreteModel()
        start_time = time.time() - 5.0  # 5 seconds ago
        first_token_time = start_time + 2.0  # 2 seconds after start
        
        with patch('time.time', return_value=start_time + 5.0):
            metrics = model.collect_basic_metrics(start_time, first_token_time)
        
        assert metrics["model"] == "TestModel"
        assert metrics["processing_time"] == 5.0
        assert metrics["audio_duration"] == 0.0
        assert metrics["time_to_audio_start"] == 2.0


class TestConfig:
    """Test configuration module."""
    
    def test_default_values(self):
        """Test default configuration values."""
        assert config.AZURE_OPENAI_API_VERSION == "2025-01-01-preview"
        assert config.SPEECH_RECOGNITION_LANGUAGE == "en-US"
        assert config.SPEECH_SYNTHESIS_VOICE == "en-US-JennyMultilingualNeural"
        assert config.GPT4O_DEPLOYMENT == "gpt-4o"
        assert config.GPT4O_REALTIME_DEPLOYMENT == "gpt-4o-realtime-preview"
        assert config.GPT4O_REALTIME_MINI_DEPLOYMENT == "gpt-4o-mini-realtime-preview"
        assert config.GPT4O_AUDIO_DEPLOYMENT == "gpt-4o-audio-preview"
        assert config.GPT41_MINI_DEPLOYMENT == "gpt-4.1-mini"
        assert config.WHISPER_DEPLOYMENT == "whisper"
        assert config.TTS_DEPLOYMENT == "tts"
        assert config.GPT4O_MINI_TTS_DEPLOYMENT == "gpt-4o-mini-tts"
        assert config.GPT4O_TRANSCRIBE_DEPLOYMENT == "gpt-4o-transcribe"
        assert config.GPT4O_MINI_TRANSCRIBE_DEPLOYMENT == "gpt-4o-mini-transcribe"
    
    def test_default_prompt(self):
        """Test the default prompt."""
        assert "Translate" in config.DEFAULT_PROMPT
        assert "Spanish" in config.DEFAULT_PROMPT
    
    def test_default_benchmark_settings(self):
        """Test default benchmark settings."""
        assert config.BENCHMARK_ITERATIONS == 3
        assert config.BENCHMARK_PAUSE_SECONDS == 10.0
        assert config.ITERATION_PAUSE_SECONDS == 1.0
        assert config.MAX_TOKENS == 1000
    
    def test_default_app_settings(self):
        """Test default application settings."""
        assert config.APP_PORT == 7860
        assert config.SHARE_APP is False
    
    @patch.dict(os.environ, {
        "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com/",
        "AZURE_OPENAI_API_KEY": "test-key",
        "SPEECH_KEY": "test-speech-key",
        "SPEECH_REGION": "test-region"
    })
    def test_environment_variables(self):
        """Test that environment variables override defaults."""
        # Need to reload the config module to pick up new env vars
        import importlib
        importlib.reload(config)
        
        assert config.AZURE_OPENAI_ENDPOINT == "https://test.openai.azure.com/"
        assert config.AZURE_OPENAI_API_KEY == "test-key"
        assert config.SPEECH_KEY == "test-speech-key"
        assert config.SPEECH_REGION == "test-region"
    
    @patch.dict(os.environ, {
        "BENCHMARK_ITERATIONS": "5",
        "BENCHMARK_PAUSE_SECONDS": "15.5",
        "ITERATION_PAUSE_SECONDS": "2.5",
        "MAX_TOKENS": "2000",
        "APP_PORT": "8080",
        "SHARE_APP": "true"
    })
    def test_numeric_environment_variables(self):
        """Test numeric environment variable parsing."""
        import importlib
        importlib.reload(config)
        
        assert config.BENCHMARK_ITERATIONS == 5
        assert config.BENCHMARK_PAUSE_SECONDS == 15.5
        assert config.ITERATION_PAUSE_SECONDS == 2.5
        assert config.MAX_TOKENS == 2000
        assert config.APP_PORT == 8080
        assert config.SHARE_APP is True
    
    @patch.dict(os.environ, {
        "BENCHMARK_ITERATIONS": "invalid",
        "BENCHMARK_PAUSE_SECONDS": "invalid",
        "ITERATION_PAUSE_SECONDS": "invalid"
    })
    def test_invalid_numeric_environment_variables(self):
        """Test handling of invalid numeric environment variables."""
        import importlib
        with patch('logging.warning') as mock_warning:
            importlib.reload(config)
            
            # Should have logged warnings and used defaults
            assert mock_warning.call_count >= 3
            assert config.BENCHMARK_ITERATIONS == 3
            assert config.BENCHMARK_PAUSE_SECONDS == 10.0
            assert config.ITERATION_PAUSE_SECONDS == 1.0
    
    @patch.dict(os.environ, {
        "MINITTS_OPENAI_ENDPOINT": "https://mini.openai.azure.com/",
        "MINITTS_OPENAI_API_KEY": "mini-key"
    })
    def test_minitts_configuration(self):
        """Test mini-TTS specific configuration."""
        import importlib
        importlib.reload(config)
        
        assert config.MINITTS_OPENAI_ENDPOINT == "https://mini.openai.azure.com/"
        assert config.MINITTS_OPENAI_API_KEY == "mini-key"
    
    @patch.dict(os.environ, {
        "SHARE_APP": "false"
    })
    def test_share_app_false(self):
        """Test SHARE_APP with false value."""
        import importlib
        importlib.reload(config)
        
        assert config.SHARE_APP is False
    
    @patch.dict(os.environ, {
        "SHARE_APP": "1"
    })
    def test_share_app_one(self):
        """Test SHARE_APP with '1' value."""
        import importlib
        importlib.reload(config)
        
        assert config.SHARE_APP is True
    
    @patch.dict(os.environ, {
        "SHARE_APP": "yes"
    })
    def test_share_app_yes(self):
        """Test SHARE_APP with 'yes' value."""
        import importlib
        importlib.reload(config)
        
        assert config.SHARE_APP is True