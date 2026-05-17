"""
TCP client for sending frame data to the Zynq device.
"""

import socket
import struct
import time
import threading
from typing import Optional, Callable

import numpy as np


class DeviceClient:
    """TCP client that sends BRAM-formatted frame data to the Zynq FPGA."""

    def __init__(
        self,
        host: str = "192.168.1.10",
        port: int = 5000,
        mock: bool = False,
        on_status_change: Optional[Callable[[str], None]] = None,
    ):
        self.host = host
        self.port = port
        self.mock = mock
        self.on_status_change = on_status_change

        self._socket: Optional[socket.socket] = None
        self._lock = threading.Lock()
        self._connected = False
        self._cycle_index = 0
        self._frames_sent = 0
        self._last_send_time = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    def connect(self) -> bool:
        """Establish TCP connection to the device."""
        if self.mock:
            self._connected = True
            self._notify_status("MOCK_CONNECTED")
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.settimeout(5.0)
            self._socket.connect((self.host, self.port))
            self._socket.settimeout(None)
            self._connected = True
            self._notify_status("CONNECTED")
            return True
        except Exception as e:
            self._connected = False
            self._notify_status(f"ERROR: {e}")
            return False

    def disconnect(self):
        """Close TCP connection."""
        if self.mock:
            self._connected = False
            self._notify_status("MOCK_DISCONNECTED")
            return

        with self._lock:
            if self._socket:
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            self._connected = False
        self._notify_status("DISCONNECTED")

    def send_frame(
        self,
        cycle_index: int,
        bram_rows: np.ndarray,
    ) -> bool:
        """Send a single frame (64 uint32 values) to the device.

        Args:
            cycle_index: 16-bit cycle index (placed in row0 [31:16])
            bram_rows: (64,) uint32 array

        Returns:
            True if sent successfully
        """
        if not self._connected:
            return False

        if bram_rows.dtype != np.uint32:
            bram_rows = bram_rows.astype(np.uint32)

        if bram_rows.shape != (64,):
            raise ValueError(f"bram_rows must be (64,), got {bram_rows.shape}")

        # Ensure row0 has the correct cycle index
        bram_rows = bram_rows.copy()
        bram_rows[0] = (bram_rows[0] & 0x0000FFFF) | ((cycle_index & 0xFFFF) << 16)

        # Pack as little-endian uint32
        data = bram_rows.tobytes()

        if self.mock:
            self._frames_sent += 1
            self._last_send_time = time.perf_counter()
            return True

        with self._lock:
            if self._socket is None:
                return False
            try:
                self._socket.sendall(data)
                self._frames_sent += 1
                self._last_send_time = time.perf_counter()
                return True
            except Exception as e:
                self._connected = False
                self._notify_status(f"SEND_ERROR: {e}")
                return False

    def get_stats(self) -> dict:
        """Return connection statistics."""
        return {
            "connected": self._connected,
            "frames_sent": self._frames_sent,
            "last_send_time": self._last_send_time,
        }

    def _notify_status(self, status: str):
        if self.on_status_change:
            try:
                self.on_status_change(status)
            except Exception:
                pass
