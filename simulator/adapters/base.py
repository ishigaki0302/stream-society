"""Abstract base class for LLM adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class LLMAdapter(ABC):
    """Abstract interface for LLM backends."""

    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        context: Dict,
        **kwargs,
    ) -> str:
        """Generate a text response from the LLM.

        Args:
            prompt: The prompt to send to the LLM.
            context: Additional context (topic, persona info, etc.).
            **kwargs: Additional backend-specific parameters.

        Returns:
            Generated response text.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM backend is reachable.

        Returns:
            True if the backend is available, False otherwise.
        """
        ...
