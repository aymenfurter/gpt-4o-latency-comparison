"""Utility for creating and managing OpenAI API clients."""

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from utils.exceptions import ModelInitializationError
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY,
    MINITTS_OPENAI_ENDPOINT,
    MINITTS_OPENAI_API_KEY
)


async def create_azure_openai_client(api_version: str, use_minitts_endpoint: bool = False) -> AsyncAzureOpenAI:
    """
    Create an AsyncAzureOpenAI client with appropriate authentication.

    Args:
        api_version: The Azure OpenAI API version to use
        use_minitts_endpoint: Whether to use the separate GPT-4o-mini-tts endpoint

    Returns:
        Configured AsyncAzureOpenAI client

    Raises:
        ModelInitializationError: If client creation fails
    """
    try:
        # Use the mini-TTS endpoint and API key if specified
        if use_minitts_endpoint:
            endpoint = MINITTS_OPENAI_ENDPOINT
            api_key = MINITTS_OPENAI_API_KEY
        else:
            endpoint = AZURE_OPENAI_ENDPOINT
            api_key = AZURE_OPENAI_API_KEY

        if api_key:
            # API key authentication
            return AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version
            )
        else:
            # Microsoft Entra ID authentication
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            return AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                azure_ad_token_provider=token_provider,
                api_version=api_version
            )
    except Exception as e:
        endpoint_type = "mini-TTS" if use_minitts_endpoint else "standard"
        raise ModelInitializationError(
            f"Failed to initialize OpenAI client ({endpoint_type} endpoint): {str(e)}") from e
