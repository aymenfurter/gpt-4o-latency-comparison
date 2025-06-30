import logging
from models.gpt4o_realtime_model import GPT4ORealtimeModel
from config import GPT4O_REALTIME_MINI_DEPLOYMENT

# Configure logger
logger = logging.getLogger(__name__)


class GPT4ORealtimeMiniModel(GPT4ORealtimeModel):
    """
    GPT-4o Realtime Mini model implementation.

    Uses the GPT-4o Realtime Mini deployment for streaming responses with audio.
    This is a lighter variant of the GPT-4o Realtime model with potentially faster response times.
    """

    def __init__(self):
        """Initialize the GPT-4o Realtime Mini model."""
        # Call parent constructor but override the name
        super().__init__()
        self.name = "GPT-4o Realtime Mini"
        self.deployment_name = GPT4O_REALTIME_MINI_DEPLOYMENT or "gpt-4o-mini-realtime-preview"
        logger.info(
            f"Initialized GPT-4o Realtime Mini with deployment: {self.deployment_name}")
