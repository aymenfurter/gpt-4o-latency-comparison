"""Utility for creating and managing OpenAI API clients."""

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from utils.exceptions import ModelInitializationError
from config import (
    AZURE_OPENAI_ENDPOINT,
    AZURE_OPENAI_API_KEY
)


async def create_azure_openai_client(api_version: str) -> AsyncAzureOpenAI:
    """
    Create an AsyncAzureOpenAI client with appropriate authentication.

    Args:
        api_version: The Azure OpenAI API version to use

    Returns:
        Configured AsyncAzureOpenAI client

    Raises:
        ModelInitializationError: If client creation fails
    """
    try:
        if AZURE_OPENAI_API_KEY:
            # API key authentication
            return AsyncAzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                api_version=api_version
            )
        else:
            # Microsoft Entra ID authentication
            credential = DefaultAzureCredential()
            token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
            return AsyncAzureOpenAI(
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_ad_token_provider=token_provider,
                api_version=api_version
            )
    except Exception as e:
        raise ModelInitializationError(
            f"Failed to initialize OpenAI client: {str(e)}") from e
