"""MMDAgent-EX WebSocket bridge stub."""
from __future__ import annotations

import logging
import os
from typing import Optional

from .base import AvatarBridge

logger = logging.getLogger(__name__)


class MMDAgentBridge(AvatarBridge):
    """Bridge to MMDAgent-EX via WebSocket.

    TODO: Full implementation requires:
    1. Install websockets package: pip install websockets
    2. Configure MMDAGENT_WS_URL environment variable
    3. Implement async WebSocket connection in connect()
    4. Implement message protocol for send_text and send_gesture
    5. Add reconnection logic with exponential backoff
    6. Handle MMDAgent-EX message format (FST plugin API)
    """

    def __init__(self) -> None:
        self._ws_url = os.environ.get("MMDAGENT_WS_URL", "ws://localhost:9000/bridge")
        self._connected = False
        self._ws = None  # TODO: websockets.WebSocketClientProtocol

    def connect(self) -> bool:
        """Attempt WebSocket connection to MMDAgent-EX.

        TODO: Implement async WebSocket connection:
            import asyncio
            import websockets
            async def _connect():
                self._ws = await websockets.connect(self._ws_url)
                return True
            return asyncio.run(_connect())
        """
        logger.warning(
            "MMDAgentBridge.connect() is a stub. "
            "WebSocket URL: %s. Returning False.",
            self._ws_url,
        )
        return False

    def send_text(self, text: str, emotion: Optional[str] = None) -> bool:
        """Send speech text to avatar.

        TODO: Implement message send:
            message = {"type": "speech", "text": text, "emotion": emotion or "neutral"}
            self._ws.send(json.dumps(message))
            return True
        """
        if not self._connected:
            logger.warning("MMDAgentBridge: not connected, cannot send text")
            return False

        # TODO: implement actual send
        logger.warning("MMDAgentBridge.send_text() is a stub. text=%r", text)
        return False

    def send_gesture(self, gesture: str) -> bool:
        """Send gesture command to avatar.

        TODO: Implement gesture message:
            message = {"type": "gesture", "name": gesture}
            self._ws.send(json.dumps(message))
            return True
        """
        if not self._connected:
            logger.warning("MMDAgentBridge: not connected, cannot send gesture")
            return False

        # TODO: implement actual send
        logger.warning("MMDAgentBridge.send_gesture() is a stub. gesture=%s", gesture)
        return False

    def is_connected(self) -> bool:
        """Return connection status."""
        return self._connected

    def disconnect(self) -> None:
        """Close WebSocket connection.

        TODO: Implement graceful close:
            if self._ws:
                asyncio.run(self._ws.close())
                self._ws = None
        """
        self._connected = False
        logger.info("MMDAgentBridge: disconnected (stub)")
