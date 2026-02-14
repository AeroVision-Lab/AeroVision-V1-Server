"""
Custom exceptions for Aerovision-V1-Server.
"""


class AerovisionException(Exception):
    """Base exception for all Aerovision errors."""

    def __init__(self, message: str, code: str = "AERO_ERROR"):
        self.message = message
        self.code = code
        super().__init__(self.message)


class ImageLoadError(AerovisionException):
    """Exception raised when image loading fails."""

    def __init__(self, message: str):
        super().__init__(message, code="IMAGE_LOAD_ERROR")


class InferenceError(AerovisionException):
    """Exception raised when inference fails."""

    def __init__(self, message: str):
        super().__init__(message, code="INFERENCE_ERROR")


class ModelNotLoadedError(AerovisionException):
    """Exception raised when model is not loaded."""

    def __init__(self, model_name: str):
        super().__init__(f"Model not loaded: {model_name}", code="MODEL_NOT_LOADED")


class ValidationError(AerovisionException):
    """Exception raised when request validation fails."""

    def __init__(self, message: str):
        super().__init__(message, code="VALIDATION_ERROR")


class RateLimitError(AerovisionException):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, code="RATE_LIMIT_ERROR")
