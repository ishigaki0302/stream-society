"""vLLM adapter stub for OpenAI-compatible API."""
from __future__ import annotations

import logging
import os
from typing import Dict

from .base import LLMAdapter

logger = logging.getLogger(__name__)


class VLLMAdapter(LLMAdapter):
    """Adapter for vLLM's OpenAI-compatible API.

    TODO: Full implementation requires:
    1. Install openai package: pip install openai
    2. Configure VLLM_BASE_URL and VLLM_MODEL environment variables
    3. Implement actual API calls in generate_response()
    4. Add streaming support for real-time responses
    5. Add retry logic with exponential backoff
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed
        self._base_url = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        self._model = os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")
        self._client = None  # TODO: initialize openai.OpenAI(base_url=..., api_key="EMPTY")

    def is_available(self) -> bool:
        """Check if vLLM endpoint is reachable.

        TODO: Implement health check:
            import urllib.request
            try:
                urllib.request.urlopen(f"{self._base_url}/models", timeout=2)
                return True
            except Exception:
                return False
        """
        logger.warning(
            "VLLMAdapter.is_available() is a stub. "
            "vLLM endpoint at %s not checked.",
            self._base_url,
        )
        return False

    def generate_response(
        self,
        prompt: str,
        context: Dict,
        **kwargs,
    ) -> str:
        """Generate response via vLLM API.

        TODO: Implement actual API call:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=kwargs.get("temperature", 0.8),
                max_tokens=kwargs.get("max_tokens", 200),
            )
            return response.choices[0].message.content

        Args:
            prompt: The prompt to send to vLLM.
            context: Additional context for the request.
            **kwargs: Additional parameters (temperature, max_tokens, etc.).

        Raises:
            NotImplementedError: Always raised in stub implementation.
        """
        raise NotImplementedError(
            "vLLM adapter not yet connected. "
            f"Set VLLM_BASE_URL={self._base_url} and ensure vLLM is running. "
            "See TODO comments in this file for implementation steps."
        )
