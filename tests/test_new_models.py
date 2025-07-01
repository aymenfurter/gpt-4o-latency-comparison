"""Tests for the new mini-TTS model implementations."""
import pytest
import time
import io
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from models.gpt4o_minitts_transcribe_model import GPT4OMiniTTSTranscribeModel
from models.gpt4o_minitts_mini_transcribe_model import GPT4OMiniTTSMiniTranscribeModel
from utils.exceptions import ModelInitializationError, ModelGenerationError


class TestGPT4OMiniTTSTranscribeModel:
    """Test the GPT4OMiniTTSTranscribeModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = GPT4OMiniTTSTranscribeModel()
        
        assert model.name == "GPT-4o + GPT-4o-transcribe + GPT-4o-mini-tts"
        assert model.gpt4o_deployment == "gpt-4o"
        assert model.tts_deployment == "gpt-4o-mini-tts"
        assert model.transcribe_deployment == "gpt-4o-transcribe"
        assert model.minitts_client is None
        assert model.initialized is False
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.create_azure_openai_client')
    async def test_initialize_same_endpoint(self, mock_create_client):
        """Test initialization when endpoints are the same."""
        model = GPT4OMiniTTSTranscribeModel()
        
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with patch('models.gpt4o_minitts_transcribe_model.MINITTS_OPENAI_ENDPOINT', 'https://same.endpoint'):
            with patch('models.gpt4o_minitts_transcribe_model.AZURE_OPENAI_ENDPOINT', 'https://same.endpoint'):
                with patch('models.gpt4o_minitts_transcribe_model.MINITTS_OPENAI_API_KEY', 'same-key'):
                    with patch('models.gpt4o_minitts_transcribe_model.AZURE_OPENAI_API_KEY', 'same-key'):
                        await model.initialize()
        
        assert model.client == mock_client
        assert model.minitts_client == mock_client  # Should reuse the same client
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.create_azure_openai_client')
    async def test_initialize_different_endpoint(self, mock_create_client):
        """Test initialization when endpoints are different."""
        model = GPT4OMiniTTSTranscribeModel()
        
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_create_client.side_effect = [mock_client1, mock_client2]
        
        with patch('models.gpt4o_minitts_transcribe_model.MINITTS_OPENAI_ENDPOINT', 'https://mini.endpoint'):
            with patch('models.gpt4o_minitts_transcribe_model.AZURE_OPENAI_ENDPOINT', 'https://main.endpoint'):
                await model.initialize()
        
        assert model.client == mock_client1
        assert model.minitts_client == mock_client2
        assert mock_create_client.call_count == 2
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.create_azure_openai_client')
    async def test_initialize_failure(self, mock_create_client):
        """Test initialization failure."""
        model = GPT4OMiniTTSTranscribeModel()
        
        mock_create_client.side_effect = Exception("Client creation failed")
        
        with pytest.raises(ModelInitializationError):
            await model.initialize()
    
    @pytest.mark.asyncio
    async def test_get_tts_headers_with_api_key(self):
        """Test _get_tts_headers with API key."""
        model = GPT4OMiniTTSTranscribeModel()
        
        with patch('models.gpt4o_minitts_transcribe_model.MINITTS_OPENAI_API_KEY', 'test-key'):
            headers = await model._get_tts_headers()
        
        assert headers['Content-Type'] == 'application/json'
        assert headers['api-key'] == 'test-key'
        assert 'Authorization' not in headers
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.DefaultAzureCredential')
    async def test_get_tts_headers_with_entra_id(self, mock_credential):
        """Test _get_tts_headers with Entra ID authentication."""
        model = GPT4OMiniTTSTranscribeModel()
        
        # Mock credential and token
        mock_token = Mock()
        mock_token.token = "test-token"
        mock_credential_instance = Mock()
        mock_credential_instance.get_token = AsyncMock(return_value=mock_token)
        mock_credential.return_value = mock_credential_instance
        
        with patch('models.gpt4o_minitts_transcribe_model.MINITTS_OPENAI_API_KEY', None):
            headers = await model._get_tts_headers()
        
        assert headers['Content-Type'] == 'application/json'
        assert headers['Authorization'] == 'Bearer test-token'
        assert 'api-key' not in headers
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.DefaultAzureCredential')
    async def test_get_tts_headers_auth_failure(self, mock_credential):
        """Test _get_tts_headers authentication failure."""
        model = GPT4OMiniTTSTranscribeModel()
        
        mock_credential.side_effect = Exception("Auth failed")
        
        with patch('models.gpt4o_minitts_transcribe_model.MINITTS_OPENAI_API_KEY', None):
            with pytest.raises(ModelInitializationError):
                await model._get_tts_headers()
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.requests.post')
    @patch('models.gpt4o_minitts_transcribe_model.get_mp3_duration')
    async def test_generate_response_from_audio_success(self, mock_get_duration, mock_post):
        """Test successful response generation."""
        model = GPT4OMiniTTSTranscribeModel()
        model.initialized = True
        
        # Mock the client
        mock_client = Mock()
        model.client = mock_client
        
        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        
        # Mock chat completion response
        mock_choice = Mock()
        mock_choice.message.content = "Response text"
        mock_usage = Mock()
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 20
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock TTS response
        mock_tts_response = Mock()
        mock_tts_response.status_code = 200
        mock_tts_response.content = b"fake_audio_data"
        mock_post.return_value = mock_tts_response
        
        # Mock duration calculation
        mock_get_duration.return_value = 2.5
        
        # Mock TTS headers
        with patch.object(model, '_get_tts_headers', new_callable=AsyncMock) as mock_headers:
            mock_headers.return_value = {'api-key': 'test-key', 'Content-Type': 'application/json'}
            
            text, metrics, audio = await model.generate_response_from_audio(b"test_audio", "test prompt")
        
        assert text == "Response text"
        assert metrics["model"] == model.name
        assert metrics["transcribe_time"] > 0
        assert metrics["text_generation_time"] > 0
        assert metrics["tts_time"] > 0
        assert metrics["audio_duration"] == 2.5
        assert metrics["token_count"] == 10
        assert metrics["total_tokens"] == 20
        assert metrics["audio_size_bytes"] == len(b"fake_audio_data")
        assert audio == b"fake_audio_data"
    
    @pytest.mark.asyncio
    async def test_generate_response_from_audio_transcription_failure(self):
        """Test response generation with transcription failure."""
        model = GPT4OMiniTTSTranscribeModel()
        model.initialized = True
        
        # Mock the client to fail on transcription
        mock_client = Mock()
        model.client = mock_client
        mock_client.audio.transcriptions.create = AsyncMock(side_effect=Exception("Transcription failed"))
        
        text, metrics, audio = await model.generate_response_from_audio(b"test_audio")
        
        assert "Error with" in text
        assert "error" in metrics
        assert audio is None
        assert metrics["audio_input_size_bytes"] == len(b"test_audio")
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_transcribe_model.requests.post')
    async def test_generate_response_from_audio_tts_failure(self, mock_post):
        """Test response generation with TTS failure."""
        model = GPT4OMiniTTSTranscribeModel()
        model.initialized = True
        
        # Mock the client
        mock_client = Mock()
        model.client = mock_client
        
        # Mock successful transcription and chat
        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        
        mock_choice = Mock()
        mock_choice.message.content = "Response text"
        mock_usage = Mock()
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 20
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock TTS failure
        mock_tts_response = Mock()
        mock_tts_response.status_code = 500
        mock_tts_response.text = "TTS Error"
        mock_post.return_value = mock_tts_response
        
        # Mock TTS headers
        with patch.object(model, '_get_tts_headers', new_callable=AsyncMock) as mock_headers:
            mock_headers.return_value = {'api-key': 'test-key', 'Content-Type': 'application/json'}
            
            text, metrics, audio = await model.generate_response_from_audio(b"test_audio")
        
        assert "Error with" in text
        assert "error" in metrics
        assert audio is None


class TestGPT4OMiniTTSMiniTranscribeModel:
    """Test the GPT4OMiniTTSMiniTranscribeModel class."""
    
    def test_init(self):
        """Test model initialization."""
        model = GPT4OMiniTTSMiniTranscribeModel()
        
        assert model.name == "GPT-4o + GPT-4o-mini-transcribe + GPT-4o-mini-tts"
        assert model.gpt4o_deployment == "gpt-4o"
        assert model.tts_deployment == "gpt-4o-mini-tts"
        assert model.transcribe_deployment == "gpt-4o-mini-transcribe"
        assert model.minitts_client is None
        assert model.initialized is False
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_mini_transcribe_model.create_azure_openai_client')
    async def test_initialize_same_endpoint(self, mock_create_client):
        """Test initialization when endpoints are the same."""
        model = GPT4OMiniTTSMiniTranscribeModel()
        
        mock_client = Mock()
        mock_create_client.return_value = mock_client
        
        with patch('models.gpt4o_minitts_mini_transcribe_model.MINITTS_OPENAI_ENDPOINT', 'https://same.endpoint'):
            with patch('models.gpt4o_minitts_mini_transcribe_model.AZURE_OPENAI_ENDPOINT', 'https://same.endpoint'):
                with patch('models.gpt4o_minitts_mini_transcribe_model.MINITTS_OPENAI_API_KEY', 'same-key'):
                    with patch('models.gpt4o_minitts_mini_transcribe_model.AZURE_OPENAI_API_KEY', 'same-key'):
                        await model.initialize()
        
        assert model.client == mock_client
        assert model.minitts_client == mock_client  # Should reuse the same client
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_mini_transcribe_model.create_azure_openai_client')
    async def test_initialize_different_endpoint(self, mock_create_client):
        """Test initialization when endpoints are different."""
        model = GPT4OMiniTTSMiniTranscribeModel()
        
        mock_client1 = Mock()
        mock_client2 = Mock()
        mock_create_client.side_effect = [mock_client1, mock_client2]
        
        with patch('models.gpt4o_minitts_mini_transcribe_model.MINITTS_OPENAI_ENDPOINT', 'https://mini.endpoint'):
            with patch('models.gpt4o_minitts_mini_transcribe_model.AZURE_OPENAI_ENDPOINT', 'https://main.endpoint'):
                await model.initialize()
        
        assert model.client == mock_client1
        assert model.minitts_client == mock_client2
        assert mock_create_client.call_count == 2
    
    @pytest.mark.asyncio
    @patch('models.gpt4o_minitts_mini_transcribe_model.requests.post')
    @patch('models.gpt4o_minitts_mini_transcribe_model.get_mp3_duration')
    async def test_generate_response_from_audio_success(self, mock_get_duration, mock_post):
        """Test successful response generation."""
        model = GPT4OMiniTTSMiniTranscribeModel()
        model.initialized = True
        
        # Mock the client
        mock_client = Mock()
        model.client = mock_client
        
        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        
        # Mock chat completion response
        mock_choice = Mock()
        mock_choice.message.content = "Response text"
        mock_usage = Mock()
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 20
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock TTS response
        mock_tts_response = Mock()
        mock_tts_response.status_code = 200
        mock_tts_response.content = b"fake_audio_data"
        mock_post.return_value = mock_tts_response
        
        # Mock duration calculation
        mock_get_duration.return_value = 2.5
        
        # Mock TTS headers
        with patch.object(model, '_get_tts_headers', new_callable=AsyncMock) as mock_headers:
            mock_headers.return_value = {'api-key': 'test-key', 'Content-Type': 'application/json'}
            
            text, metrics, audio = await model.generate_response_from_audio(b"test_audio", "test prompt")
        
        assert text == "Response text"
        assert metrics["model"] == model.name
        assert metrics["transcribe_time"] > 0
        assert metrics["text_generation_time"] > 0
        assert metrics["tts_time"] > 0
        assert metrics["audio_duration"] == 2.5
        assert metrics["token_count"] == 10
        assert metrics["total_tokens"] == 20
        assert metrics["audio_size_bytes"] == len(b"fake_audio_data")
        assert audio == b"fake_audio_data"
        
        # Check that whisper_time is set for consistency
        assert metrics["whisper_time"] == metrics["transcribe_time"]
    
    @pytest.mark.asyncio
    async def test_generate_response_from_audio_without_text_prompt(self):
        """Test response generation without additional text prompt."""
        model = GPT4OMiniTTSMiniTranscribeModel()
        model.initialized = True
        
        # Mock the client
        mock_client = Mock()
        model.client = mock_client
        
        # Mock transcription response
        mock_transcription = Mock()
        mock_transcription.text = "Hello world"
        mock_client.audio.transcriptions.create = AsyncMock(return_value=mock_transcription)
        
        # Mock chat completion response
        mock_choice = Mock()
        mock_choice.message.content = "Response text"
        mock_usage = Mock()
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 20
        mock_response = Mock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        
        # Mock other components
        with patch('models.gpt4o_minitts_mini_transcribe_model.requests.post') as mock_post:
            mock_tts_response = Mock()
            mock_tts_response.status_code = 200
            mock_tts_response.content = b"fake_audio_data"
            mock_post.return_value = mock_tts_response
            
            with patch('models.gpt4o_minitts_mini_transcribe_model.get_mp3_duration', return_value=2.5):
                with patch.object(model, '_get_tts_headers', new_callable=AsyncMock) as mock_headers:
                    mock_headers.return_value = {'api-key': 'test-key', 'Content-Type': 'application/json'}
                    
                    text, metrics, audio = await model.generate_response_from_audio(b"test_audio")
        
        # Verify that chat completion was called with the transcription only
        args, kwargs = mock_client.chat.completions.create.call_args
        messages = kwargs['messages']
        assert len(messages) == 1
        assert "Respond to this transcribed audio: Hello world" in messages[0]["content"]