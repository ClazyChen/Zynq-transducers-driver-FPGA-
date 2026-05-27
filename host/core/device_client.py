"""
TCP client for sending frame data to the Zynq device.
"""

import socket
import time
import threading
from typing import Optional, Callable

import numpy as np

# Large send buffer helps sustain ~10 MB/s bursts on Windows
TCP_SNDBUF_SIZE = 1 << 20


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
        if self._connected:
            self.disconnect()

        if self.mock:
            self._connected = True
            self._notify_status("MOCK_CONNECTED")
            return True

        try:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._socket.setsockopt(
                socket.SOL_SOCKET, socket.SO_SNDBUF, TCP_SNDBUF_SIZE
            )
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
        """Close TCP connection so PS can be reflashed and PC can reconnect."""
        was_mock = self.mock
        with self._lock:
            if self._socket:
                try:
                    self._socket.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    self._socket.close()
                except Exception:
                    pass
                self._socket = None
            self._connected = False
        if was_mock:
            self._notify_status("MOCK_DISCONNECTED")
        else:
            self._notify_status("DISCONNECTED")

    def send_burst(self, bram_rows_batch: np.ndarray) -> bool:
        """Send multiple frames in one TCP write.

        Args:
            bram_rows_batch: (N, 64) uint32 array; row0 of each frame must
                already contain the correct cycle_index in [31:16].

        Returns:
            True if sent successfully
        """
        if not self._connected:
            return False

        if bram_rows_batch.dtype != np.uint32:
            bram_rows_batch = bram_rows_batch.astype(np.uint32)

        if bram_rows_batch.ndim != 2 or bram_rows_batch.shape[1] != 64:
            raise ValueError(
                f"bram_rows_batch must be (N, 64), got {bram_rows_batch.shape}"
            )

        n = bram_rows_batch.shape[0]
        if n == 0:
            return True

        data = bram_rows_batch.tobytes()

        if self.mock:
            self._frames_sent += n
            self._last_send_time = time.perf_counter()
            return True

        with self._lock:
            if self._socket is None:
                return False
            try:
                self._socket.sendall(data)
                self._frames_sent += n
                self._last_send_time = time.perf_counter()
                return True
            except Exception as e:
                self._connected = False
                self._notify_status(f"SEND_ERROR: {e}")
                return False

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
        if bram_rows.dtype != np.uint32:
            bram_rows = bram_rows.astype(np.uint32)

        if bram_rows.shape != (64,):
            raise ValueError(f"bram_rows must be (64,), got {bram_rows.shape}")

        bram_rows = bram_rows.copy()
        bram_rows[0] = (bram_rows[0] & 0x0000FFFF) | ((cycle_index & 0xFFFF) << 16)
        return self.send_burst(bram_rows.reshape(1, 64))

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
