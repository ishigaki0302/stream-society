"""Abstract base class for avatar bridges."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class AvatarBridge(ABC):
    """Abstract interface for avatar communication bridges."""

    @abstractmethod
    def connect(self) -> bool:
        """Establish connection to the avatar system.

        Returns:
            True if connection was established successfully, False otherwise.
        """
        ...

    @abstractmethod
    def send_text(self, text: str, emotion: Optional[str] = None) -> bool:
        """Send text (speech) to the avatar.

        Args:
            text: Text to speak.
            emotion: Optional emotion tag (happy, sad, surprised, etc.).

        Returns:
            True if sent successfully, False otherwise.
        """
        ...

    @abstractmethod
    def send_gesture(self, gesture: str) -> bool:
        """Send a gesture command to the avatar.

        Args:
            gesture: Gesture name (wave, nod, point, etc.).

        Returns:
            True if sent successfully, False otherwise.
        """
        ...

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the bridge is currently connected.

        Returns:
            True if connected, False otherwise.
        """
        ...

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from the avatar system."""
        ...
