"""Custom exceptions for the GPT-4o latency comparison project."""


class BenchmarkError(Exception):
    """Base class for all benchmark-related exceptions."""
    pass


class ModelInitializationError(BenchmarkError):
    """Raised when a model fails to initialize."""
    pass


class ModelGenerationError(BenchmarkError):
    """Raised when a model fails to generate a response."""
    pass


class AudioProcessingError(BenchmarkError):
    """Raised when audio processing fails."""
    pass


class ConfigurationError(BenchmarkError):
    """Raised when a configuration issue is detected."""
    pass
