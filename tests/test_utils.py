"""Tests for utility functions."""
import pytest
import time
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, mock_open
from utils.metrics import (
    calculate_tokens_per_second, 
    create_error_metrics, 
    calculate_relative_performance,
    format_metric_value
)
from utils.exceptions import (
    BenchmarkError, 
    ModelGenerationError, 
    ModelInitializationError,
    AudioProcessingError,
    ConfigurationError
)
from utils.audio_utils import (
    convert_pcm_to_wav,
    get_wav_duration,
    get_mp3_duration
)
from utils.client import create_azure_openai_client


class TestMetrics:
    """Test metrics utility functions."""
    
    def test_calculate_tokens_per_second_normal(self):
        """Test normal token calculation."""
        tokens = 100
        generation_time = 10.0
        result = calculate_tokens_per_second(tokens, generation_time)
        assert result == 10.0
    
    def test_calculate_tokens_per_second_zero_time(self):
        """Test token calculation with zero time."""
        tokens = 100
        generation_time = 0.0
        result = calculate_tokens_per_second(tokens, generation_time)
        assert result == 0.0
    
    def test_calculate_tokens_per_second_negative_time(self):
        """Test token calculation with negative time."""
        tokens = 100
        generation_time = -1.0
        result = calculate_tokens_per_second(tokens, generation_time)
        assert result == 0.0
    
    def test_calculate_tokens_per_second_zero_tokens(self):
        """Test token calculation with zero tokens."""
        tokens = 0
        generation_time = 10.0
        result = calculate_tokens_per_second(tokens, generation_time)
        assert result == 0.0
    
    def test_calculate_tokens_per_second_negative_tokens(self):
        """Test token calculation with negative tokens."""
        tokens = -10
        generation_time = 10.0
        result = calculate_tokens_per_second(tokens, generation_time)
        assert result == 0.0

    def test_create_error_metrics(self):
        """Test error metrics creation."""
        model_name = "test-model"
        error = Exception("test error")
        start_time = time.time() - 5.0  # 5 seconds ago
        
        with patch('time.time', return_value=start_time + 5.0):
            metrics = create_error_metrics(model_name, error, start_time)
        
        assert metrics["model"] == model_name
        assert metrics["error"] == "test error"
        assert metrics["processing_time"] == 5.0
        assert metrics["time_to_audio_start"] == 0
        assert metrics["audio_duration"] == 0
        assert metrics["token_count"] == 0
        assert metrics["total_tokens"] == 0
        assert metrics["tokens_per_second"] == 0
        assert metrics["audio_size_bytes"] == 0

    def test_calculate_relative_performance_lower_is_better(self):
        """Test relative performance calculation for lower-is-better metrics."""
        df = pd.DataFrame({
            'latency': [1.0, 2.0, 3.0, 1.5]
        })
        
        result = calculate_relative_performance(df, 'latency', higher_is_better=False)
        
        # Best value is 1.0, so relative performance should be (1.0/value) * 100
        expected = pd.Series([100.0, 50.0, 33.333333, 66.666667], name='latency')
        pd.testing.assert_series_equal(result, expected, check_dtype=False, rtol=1e-5)
    
    def test_calculate_relative_performance_higher_is_better(self):
        """Test relative performance calculation for higher-is-better metrics."""
        df = pd.DataFrame({
            'tokens_per_second': [10.0, 20.0, 15.0, 5.0]
        })
        
        result = calculate_relative_performance(df, 'tokens_per_second', higher_is_better=True)
        
        # Best value is 20.0, so relative performance should be (value/20.0) * 100
        expected = pd.Series([50.0, 100.0, 75.0, 25.0], name='tokens_per_second')
        pd.testing.assert_series_equal(result, expected, check_dtype=False)
    
    def test_calculate_relative_performance_empty_df(self):
        """Test relative performance calculation with empty DataFrame."""
        df = pd.DataFrame()
        result = calculate_relative_performance(df, 'latency')
        assert result.empty
    
    def test_calculate_relative_performance_missing_column(self):
        """Test relative performance calculation with missing column."""
        df = pd.DataFrame({'other': [1, 2, 3]})
        result = calculate_relative_performance(df, 'latency')
        assert result.empty
    
    def test_calculate_relative_performance_with_nan(self):
        """Test relative performance calculation with NaN values."""
        df = pd.DataFrame({
            'latency': [1.0, np.nan, 3.0, 1.5]
        })
        
        result = calculate_relative_performance(df, 'latency', higher_is_better=False)
        
        # Best value is 1.0, NaN should remain NaN
        expected = pd.Series([100.0, np.nan, 33.333333, 66.666667], name='latency')
        pd.testing.assert_series_equal(result, expected, check_dtype=False, rtol=1e-5)

    def test_format_metric_value_time(self):
        """Test formatting time values."""
        assert format_metric_value(1.23456, "time") == "1.235s"
        assert format_metric_value(0.001, "time") == "0.001s"
    
    def test_format_metric_value_tokens_per_second(self):
        """Test formatting tokens per second values."""
        assert format_metric_value(12.345, "tokens_per_second") == "12.3"
        assert format_metric_value(100.9, "tokens_per_second") == "100.9"
    
    def test_format_metric_value_bytes(self):
        """Test formatting byte values."""
        assert format_metric_value(1024, "bytes") == "1.0 KB"
        assert format_metric_value(2048, "bytes") == "2.0 KB"
        assert format_metric_value(1536, "bytes") == "1.5 KB"
    
    def test_format_metric_value_default(self):
        """Test formatting default values."""
        assert format_metric_value(42, "other") == "42"
        assert format_metric_value(42.123, "other") == "42.12"
    
    def test_format_metric_value_none(self):
        """Test formatting None values."""
        assert format_metric_value(None, "time") == "Not applicable"
        assert format_metric_value(pd.NA, "time") == "Not applicable"
    
    def test_format_metric_value_non_numeric(self):
        """Test formatting non-numeric values."""
        assert format_metric_value("test", "time") == "test"
        assert format_metric_value(["list"], "time") == "['list']"


class TestExceptions:
    """Test custom exception classes."""
    
    def test_benchmark_error(self):
        """Test BenchmarkError exception."""
        error = BenchmarkError("test message")
        assert str(error) == "test message"
        assert isinstance(error, Exception)
    
    def test_model_generation_error(self):
        """Test ModelGenerationError exception."""
        error = ModelGenerationError("generation failed")
        assert str(error) == "generation failed"
        assert isinstance(error, Exception)
        assert isinstance(error, BenchmarkError)
    
    def test_model_initialization_error(self):
        """Test ModelInitializationError exception."""
        error = ModelInitializationError("init failed")
        assert str(error) == "init failed"
        assert isinstance(error, Exception)
        assert isinstance(error, BenchmarkError)
    
    def test_audio_processing_error(self):
        """Test AudioProcessingError exception."""
        error = AudioProcessingError("audio failed")
        assert str(error) == "audio failed"
        assert isinstance(error, Exception)
        assert isinstance(error, BenchmarkError)
    
    def test_configuration_error(self):
        """Test ConfigurationError exception."""
        error = ConfigurationError("config failed")
        assert str(error) == "config failed"
        assert isinstance(error, Exception)
        assert isinstance(error, BenchmarkError)


class TestAudioUtils:
    """Test audio utility functions."""
    
    @patch('wave.open')
    def test_convert_pcm_to_wav(self, mock_wave_open):
        """Test PCM to WAV conversion."""
        mock_wav_file = Mock()
        mock_wave_open.return_value.__enter__.return_value = mock_wav_file
        
        pcm_data = b"fake_pcm_data"
        result = convert_pcm_to_wav(pcm_data, 48000)
        
        # Check that wave file was configured correctly
        mock_wav_file.setnchannels.assert_called_once_with(1)
        mock_wav_file.setsampwidth.assert_called_once_with(2)
        mock_wav_file.setframerate.assert_called_once_with(48000)
        mock_wav_file.writeframes.assert_called_once_with(pcm_data)
    
    @patch('wave.open')
    def test_get_wav_duration_success(self, mock_wave_open):
        """Test WAV duration calculation success."""
        mock_wav_file = Mock()
        mock_wav_file.getnframes.return_value = 48000
        mock_wav_file.getframerate.return_value = 24000
        mock_wave_open.return_value.__enter__.return_value = mock_wav_file
        
        wav_data = b"fake_wav_data"
        duration = get_wav_duration(wav_data)
        
        assert duration == 2.0  # 48000 frames / 24000 rate = 2 seconds
    
    @patch('wave.open')
    @patch('pydub.AudioSegment.from_wav')
    def test_get_wav_duration_fallback_to_pydub(self, mock_from_wav, mock_wave_open):
        """Test WAV duration calculation fallback to pydub."""
        # Wave module fails
        mock_wave_open.side_effect = Exception("Wave error")
        
        # Pydub succeeds
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=5000)  # 5 seconds in milliseconds
        mock_from_wav.return_value = mock_audio
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.wav'
            
            wav_data = b"fake_wav_data"
            duration = get_wav_duration(wav_data)
            
            assert duration == 5.0
    
    @patch('wave.open')
    @patch('pydub.AudioSegment.from_wav')
    def test_get_wav_duration_fallback_estimation(self, mock_from_wav, mock_wave_open):
        """Test WAV duration calculation fallback to estimation."""
        # Both wave and pydub fail
        mock_wave_open.side_effect = Exception("Wave error")
        mock_from_wav.side_effect = Exception("Pydub error")
        
        wav_data = b"fake_wav_data" * 10000  # Make it reasonably sized
        duration = get_wav_duration(wav_data)
        
        # Should fall back to estimation
        assert duration > 0
        assert duration <= 180  # Max duration cap
    
    def test_get_wav_duration_invalid_range(self):
        """Test WAV duration calculation with invalid duration range."""
        with patch('wave.open') as mock_wave_open:
            mock_wav_file = Mock()
            mock_wav_file.getnframes.return_value = 1000
            mock_wav_file.getframerate.return_value = 2000
            mock_wave_open.return_value.__enter__.return_value = mock_wav_file
            
            wav_data = b"fake_wav_data"
            duration = get_wav_duration(wav_data)
            
            # Duration is 0.5s which is < 1, so should try fallback
            assert duration >= 0
    
    @patch('subprocess.run')
    def test_get_mp3_duration_ffprobe_success(self, mock_run):
        """Test MP3 duration calculation with ffprobe success."""
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "10.5\n"
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.mp3'
            
            mp3_data = b"fake_mp3_data"
            duration = get_mp3_duration(mp3_data)
            
            assert duration == 10.5
    
    @patch('subprocess.run')
    @patch('pydub.AudioSegment.from_mp3')
    def test_get_mp3_duration_fallback_to_pydub(self, mock_from_mp3, mock_run):
        """Test MP3 duration calculation fallback to pydub."""
        # ffprobe fails
        mock_run.return_value.returncode = 1
        
        # Pydub succeeds
        mock_audio = Mock()
        mock_audio.__len__ = Mock(return_value=7500)  # 7.5 seconds in milliseconds
        mock_from_mp3.return_value = mock_audio
        
        with patch('tempfile.NamedTemporaryFile') as mock_temp:
            mock_temp.return_value.__enter__.return_value.name = '/tmp/test.mp3'
            
            mp3_data = b"fake_mp3_data"
            duration = get_mp3_duration(mp3_data)
            
            assert duration == 7.5
    
    @patch('subprocess.run')
    @patch('pydub.AudioSegment.from_mp3')
    def test_get_mp3_duration_estimation_fallback(self, mock_from_mp3, mock_run):
        """Test MP3 duration calculation fallback to estimation."""
        # Both ffprobe and pydub fail
        mock_run.side_effect = Exception("ffprobe error")
        mock_from_mp3.side_effect = Exception("pydub error")
        
        mp3_data = b"x" * (64 * 1024)  # 64KB 
        duration = get_mp3_duration(mp3_data)
        
        # Should estimate based on file size: 64KB / 8KB per second = 8 seconds
        # The function has min(180, max(5, estimated_duration))
        expected = 8.0  # 64KB / 8KB per second
        assert duration == expected


class TestClient:
    """Test client utility functions."""
    
    @patch('utils.client.AsyncAzureOpenAI')
    @patch('utils.client.AZURE_OPENAI_API_KEY', 'test-key')
    @patch('utils.client.AZURE_OPENAI_ENDPOINT', 'https://test.openai.azure.com/')
    @pytest.mark.asyncio
    async def test_create_azure_openai_client_with_api_key(self, mock_client):
        """Test creating client with API key."""
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        result = await create_azure_openai_client("2023-05-15")
        
        mock_client.assert_called_once_with(
            api_key='test-key',
            azure_endpoint='https://test.openai.azure.com/',
            api_version="2023-05-15"
        )
        assert result == mock_instance
    
    @patch('utils.client.AsyncAzureOpenAI')
    @patch('utils.client.MINITTS_OPENAI_API_KEY', 'mini-key')
    @patch('utils.client.MINITTS_OPENAI_ENDPOINT', 'https://mini.openai.azure.com/')
    @pytest.mark.asyncio
    async def test_create_azure_openai_client_minitts_endpoint(self, mock_client):
        """Test creating client with mini-TTS endpoint."""
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        
        result = await create_azure_openai_client("2023-05-15", use_minitts_endpoint=True)
        
        mock_client.assert_called_once_with(
            api_key='mini-key',
            azure_endpoint='https://mini.openai.azure.com/',
            api_version="2023-05-15"
        )
        assert result == mock_instance
    
    @patch('utils.client.AsyncAzureOpenAI')
    @patch('utils.client.DefaultAzureCredential')
    @patch('utils.client.get_bearer_token_provider')
    @patch('utils.client.AZURE_OPENAI_API_KEY', None)
    @patch('utils.client.AZURE_OPENAI_ENDPOINT', 'https://test.openai.azure.com/')
    @pytest.mark.asyncio
    async def test_create_azure_openai_client_with_entra_id(self, mock_token_provider, mock_credential, mock_client):
        """Test creating client with Entra ID authentication."""
        mock_instance = Mock()
        mock_client.return_value = mock_instance
        mock_credential_instance = Mock()
        mock_credential.return_value = mock_credential_instance
        mock_provider = Mock()
        mock_token_provider.return_value = mock_provider
        
        result = await create_azure_openai_client("2023-05-15")
        
        mock_credential.assert_called_once()
        mock_token_provider.assert_called_once_with(
            mock_credential_instance, "https://cognitiveservices.azure.com/.default"
        )
        mock_client.assert_called_once_with(
            azure_endpoint='https://test.openai.azure.com/',
            azure_ad_token_provider=mock_provider,
            api_version="2023-05-15"
        )
        assert result == mock_instance
    
    @patch('utils.client.AsyncAzureOpenAI')
    @pytest.mark.asyncio
    async def test_create_azure_openai_client_failure(self, mock_client):
        """Test client creation failure."""
        mock_client.side_effect = Exception("Client creation failed")
        
        with pytest.raises(ModelInitializationError) as exc_info:
            await create_azure_openai_client("2023-05-15")
        
        assert "Failed to initialize OpenAI client (standard endpoint)" in str(exc_info.value)
    
    @patch('utils.client.AsyncAzureOpenAI')
    @pytest.mark.asyncio
    async def test_create_azure_openai_client_minitts_failure(self, mock_client):
        """Test mini-TTS client creation failure."""
        mock_client.side_effect = Exception("Client creation failed")
        
        with pytest.raises(ModelInitializationError) as exc_info:
            await create_azure_openai_client("2023-05-15", use_minitts_endpoint=True)
        
        assert "Failed to initialize OpenAI client (mini-TTS endpoint)" in str(exc_info.value)