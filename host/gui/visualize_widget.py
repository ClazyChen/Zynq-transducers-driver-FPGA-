"""
Acoustic field intensity visualization widget (heatmap).
"""

import numpy as np

from PySide6.QtWidgets import QWidget, QVBoxLayout
from PySide6.QtCore import Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib

# Set font for Chinese characters on Windows
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # Fix minus sign display


class FieldVisualizationWidget(QWidget):
    """Displays the real-time acoustic field intensity as a heatmap."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        self._figure = Figure(figsize=(4, 4), dpi=100)
        self._canvas = FigureCanvas(self._figure)
        self._ax = self._figure.add_subplot(111)

        layout.addWidget(self._canvas)

        # Initialize with empty data
        self._im = self._ax.imshow(
            np.zeros((128, 128)),
            origin="upper",
            extent=[0, 128, 0, 128],
            cmap="hot",
            vmin=0,
            vmax=1,
        )
        self._ax.set_xlabel("X (mm)")
        self._ax.set_ylabel("Y (mm)")
        self._ax.set_title("声场强度")
        self._figure.colorbar(self._im, ax=self._ax, label="强度")
        self._canvas.draw()

    def update_field(self, intensity: np.ndarray):
        """Update the heatmap with new intensity data.

        Args:
            intensity: (128, 128) numpy array
        """
        intensity = np.asarray(intensity)
        if intensity.shape != (128, 128):
            return

        self._im.set_data(intensity)
        self._im.set_clim(vmin=intensity.min(), vmax=intensity.max())
        self._canvas.draw_idle()
