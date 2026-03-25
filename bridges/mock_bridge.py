"""Mock avatar bridge for testing and simulation."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from .base import AvatarBridge

logger = logging.getLogger(__name__)


class MockAvatarBridge(AvatarBridge):
    """Mock bridge that logs all calls and always returns True."""

    def __init__(self) -> None:
        self._connected = False
        self.log: List[Dict] = []

    def connect(self) -> bool:
        """Simulate connection (always succeeds)."""
        self._connected = True
        entry = {"action": "connect", "result": True}
        self.log.append(entry)
        logger.info("MockAvatarBridge: connected")
        return True

    def send_text(self, text: str, emotion: Optional[str] = None) -> bool:
        """Log text send (always succeeds)."""
        entry = {"action": "send_text", "text": text, "emotion": emotion, "result": True}
        self.log.append(entry)
        logger.info("MockAvatarBridge: send_text text=%r emotion=%s", text, emotion)
        return True

    def send_gesture(self, gesture: str) -> bool:
        """Log gesture send (always succeeds)."""
        entry = {"action": "send_gesture", "gesture": gesture, "result": True}
        self.log.append(entry)
        logger.info("MockAvatarBridge: send_gesture gesture=%s", gesture)
        return True

    def is_connected(self) -> bool:
        """Return connection status."""
        return self._connected

    def disconnect(self) -> None:
        """Simulate disconnection."""
        self._connected = False
        entry = {"action": "disconnect"}
        self.log.append(entry)
        logger.info("MockAvatarBridge: disconnected")

    def get_log(self) -> List[Dict]:
        """Return the full message log."""
        return self.log.copy()

    def clear_log(self) -> None:
        """Clear the message log."""
        self.log.clear()
