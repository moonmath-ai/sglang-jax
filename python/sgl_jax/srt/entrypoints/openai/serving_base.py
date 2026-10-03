import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any

from fastapi import Request
from fastapi.responses import ORJSONResponse, StreamingResponse

from sgl_jax.srt.entrypoints.openai.protocol import ErrorResponse, OpenAIServingRequest
from sgl_jax.srt.managers.io_struct import GenerateReqInput
from sgl_jax.srt.managers.tokenizer_manager import TokenizerManager

logger = logging.getLogger(__name__)


# Sampling fields a request may leave to the model's generation_config.json.
_MODEL_DEFAULT_SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p")


def load_model_sampling_defaults(model_path: str, preferred_json: str | None) -> dict[str, Any]:
    """Model-recommended sampling values: generation_config.json (read as written, since
    transformers' GenerationConfig hides values equal to its own defaults, e.g.
    temperature 1.0), then --preferred-sampling-params on top."""
    defaults: dict[str, Any] = {}
    path = os.path.join(model_path, "generation_config.json")
    if os.path.isfile(path):
        try:
            with open(path) as f:
                cfg = json.load(f)
            defaults = {k: cfg[k] for k in _MODEL_DEFAULT_SAMPLING_FIELDS if cfg.get(k) is not None}
        except (OSError, ValueError) as e:
            logger.warning("Ignoring unreadable %s: %s", path, e)
    if preferred_json:
        preferred = json.loads(preferred_json)
        defaults.update({k: v for k, v in preferred.items() if k in _MODEL_DEFAULT_SAMPLING_FIELDS})
    return defaults


# Base class for specific endpoint handlers
class OpenAIServingBase(ABC):
    """Abstract base class for OpenAI endpoint handlers"""

    def __init__(self, tokenizer_manager: TokenizerManager):
        self.tokenizer_manager = tokenizer_manager

    def _apply_model_sampling_defaults(self, request, sampling_params: dict[str, Any]):
        """Sampling fields the client did not send take the model's recommended values
        (upstream sglang's model sampling defaults) instead of the protocol defaults."""
        defaults = getattr(self, "_model_sampling_defaults", None)
        if defaults is None:
            args = self.tokenizer_manager.server_args
            defaults = load_model_sampling_defaults(
                args.model_path, getattr(args, "preferred_sampling_params", None)
            )
            self._model_sampling_defaults = defaults
        for key, value in defaults.items():
            if key not in request.model_fields_set:
                sampling_params[key] = value
        return sampling_params

    async def handle_request(
        self, request: OpenAIServingRequest, raw_request: Request
    ) -> Any | StreamingResponse | ErrorResponse:
        """Handle the specific request type with common pattern"""
        try:
            # Validate request
            error_msg = self._validate_request(request)
            if error_msg:
                return self.create_error_response(error_msg)

            # Convert to internal format
            adapted_request, processed_request = self._convert_to_internal_request(request)

            # Note(Xinyuan): raw_request below is only used for detecting the connection of the client
            if hasattr(request, "stream") and request.stream:
                return await self._handle_streaming_request(
                    adapted_request, processed_request, raw_request
                )
            else:
                return await self._handle_non_streaming_request(
                    adapted_request, processed_request, raw_request
                )

        except Exception as e:
            logger.exception("Error in request: %s", e)
            return self.create_error_response(
                message=f"Internal server error: {str(e)}",
                err_type="InternalServerError",
                status_code=500,
            )

    @abstractmethod
    def _request_id_prefix(self) -> str:
        """Generate request ID based on request type"""
        pass

    def _generate_request_id_base(self, request: OpenAIServingRequest) -> str | None:
        """Generate request ID based on request type"""
        return None

        # Temporarily return None in this function until the rid logic is clear.
        if rid := getattr(request, "rid", None):
            return rid

        return f"{self._request_id_prefix()}{uuid.uuid4().hex}"

    @abstractmethod
    def _convert_to_internal_request(
        self,
        request: OpenAIServingRequest,
    ) -> tuple[GenerateReqInput, OpenAIServingRequest]:
        """Convert OpenAI request to internal format"""
        pass

    async def _handle_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> StreamingResponse | ErrorResponse | ORJSONResponse:
        """Handle streaming request

        Override this method in child classes that support streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    async def _handle_non_streaming_request(
        self,
        adapted_request: GenerateReqInput,
        request: OpenAIServingRequest,
        raw_request: Request,
    ) -> Any | ErrorResponse | ORJSONResponse:
        """Handle non-streaming request

        Override this method in child classes that support non-streaming requests.
        """
        return self.create_error_response(
            message=f"{self.__class__.__name__} does not support non-streaming requests",
            err_type="NotImplementedError",
            status_code=501,
        )

    def _validate_request(self, _: OpenAIServingRequest) -> str | None:
        """Validate request"""
        return None

    def create_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: str | None = None,
    ) -> ORJSONResponse:
        """Create an error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=param,
            code=status_code,
        )
        return ORJSONResponse(content=error.model_dump(), status_code=status_code)

    def create_streaming_error_response(
        self,
        message: str,
        err_type: str = "BadRequestError",
        status_code: int = 400,
    ) -> str:
        """Create a streaming error response"""
        error = ErrorResponse(
            object="error",
            message=message,
            type=err_type,
            param=None,
            code=status_code,
        )
        return json.dumps({"error": error.model_dump()})
