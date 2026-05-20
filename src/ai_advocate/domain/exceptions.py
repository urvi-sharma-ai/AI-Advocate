class AIAdvocateError(Exception):
    """Base error for the application."""


class ConfigurationError(AIAdvocateError):
    """Missing or invalid configuration (e.g. API key)."""


class IngestionError(AIAdvocateError):
    """Failed to load or index documents."""
