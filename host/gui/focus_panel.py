"""
Focus configuration panel with 2D visualization.
"""

from typing import List, Tuple

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGridLayout, QButtonGroup, QRadioButton, QGroupBox,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont


class FocusCanvas(QWidget):
    """Custom widget that visualizes the 128mm x 128mm field and transducer array."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.foci: List[Tuple[float, float]] = []
        self._scale = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

    def set_foci(self, foci: List[Tuple[float, float]]):
        self.foci = list(foci)
        self.update()

    def _world_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates (mm, origin at top-left) to screen coordinates."""
        sx = self._offset_x + x * self._scale
        sy = self._offset_y + y * self._scale
        return sx, sy

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        w, h = self.width(), self.height()
        margin = 20

        # Compute scale to fit 128x128 mm into the widget with margin
        avail_w = w - 2 * margin
        avail_h = h - 2 * margin
        self._scale = min(avail_w / 128.0, avail_h / 128.0)

        # Center the field
        field_w = 128.0 * self._scale
        field_h = 128.0 * self._scale
        self._offset_x = (w - field_w) / 2.0
        self._offset_y = (h - field_h) / 2.0

        # Draw background
        painter.fillRect(self.rect(), QColor(240, 240, 240))

        # Draw field border
        painter.setPen(QPen(QColor(0, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        x0, y0 = self._world_to_screen(0, 0)
        x1, y1 = self._world_to_screen(128, 128)
        painter.drawRect(int(x0), int(y0), int(x1 - x0), int(y1 - y0))

        # Draw center region (80x80 mm)
        painter.setPen(QPen(QColor(180, 180, 180), 1, Qt.DashLine))
        cx0, cy0 = self._world_to_screen(24, 24)
        cx1, cy1 = self._world_to_screen(104, 104)
        painter.drawRect(int(cx0), int(cy0), int(cx1 - cx0), int(cy1 - cy0))

        # Draw transducer array (8x8, 10mm spacing)
        # Array center at (64, 64), elements from (64-35, 64-35) to (64+35, 64+35)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setBrush(QBrush(QColor(200, 200, 200)))
        element_radius = max(2, int(2.5 * self._scale))
        for row in range(8):
            for col in range(8):
                ex = 64.0 + (col - 3.5) * 10.0
                ey = 64.0 + (row - 3.5) * 10.0
                sx, sy = self._world_to_screen(ex, ey)
                painter.drawEllipse(
                    int(sx - element_radius), int(sy - element_radius),
                    element_radius * 2, element_radius * 2
                )

        # Draw foci
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0, 180)))
        focus_radius = max(4, int(4.68 * self._scale))
        for i, (fx, fy) in enumerate(self.foci):
            sx, sy = self._world_to_screen(fx, fy)
            painter.drawEllipse(
                int(sx - focus_radius), int(sy - focus_radius),
                focus_radius * 2, focus_radius * 2
            )
            # Draw label
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            font = QFont()
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(int(sx + focus_radius + 3), int(sy), f"焦{i+1}")
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0, 180)))

        # Draw axes labels
        painter.setPen(QPen(QColor(0, 0, 0), 1))
        font = QFont()
        font.setPointSize(8)
        painter.setFont(font)
        painter.drawText(int(x0), int(y0) - 5, "0,0")
        painter.drawText(int(x1) - 25, int(y0) - 5, "128,0")
        painter.drawText(int(x0), int(y1) + 12, "0,128")
        painter.drawText(int(x1) - 35, int(y1) + 12, "128,128")

        painter.end()


class FocusPanel(QWidget):
    """Panel for focus position configuration and 2D visualization."""

    foci_changed = Signal(list)  # List[Tuple[float, float]]
    mode_changed = Signal(str)   # "static" or "dynamic"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._foci: List[Tuple[float, float]] = []
        self._mode = "static"
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Mode selection
        mode_group = QGroupBox("渲染模式")
        mode_layout = QHBoxLayout()
        self._mode_buttons = QButtonGroup(self)
        self._rb_static = QRadioButton("静态（同时渲染）")
        self._rb_dynamic = QRadioButton("动态（循环切换）")
        self._rb_static.setChecked(True)
        self._mode_buttons.addButton(self._rb_static, 0)
        self._mode_buttons.addButton(self._rb_dynamic, 1)
        self._mode_buttons.idClicked.connect(self._on_mode_changed)
        mode_layout.addWidget(self._rb_static)
        mode_layout.addWidget(self._rb_dynamic)
        mode_group.setLayout(mode_layout)
        layout.addWidget(mode_group)

        # Focus list
        foci_group = QGroupBox("焦点位置（最多3个）")
        foci_layout = QGridLayout()
        foci_layout.addWidget(QLabel("X (mm)"), 0, 1)
        foci_layout.addWidget(QLabel("Y (mm)"), 0, 2)

        self._focus_edits: List[Tuple[QLineEdit, QLineEdit]] = []
        for i in range(3):
            label = QLabel(f"焦点 {i+1}:")
            x_edit = QLineEdit()
            x_edit.setPlaceholderText("64.0")
            y_edit = QLineEdit()
            y_edit.setPlaceholderText("64.0")
            x_edit.setEnabled(i == 0)  # First focus enabled by default
            y_edit.setEnabled(i == 0)
            x_edit.textChanged.connect(self._on_foci_edited)
            y_edit.textChanged.connect(self._on_foci_edited)
            foci_layout.addWidget(label, i + 1, 0)
            foci_layout.addWidget(x_edit, i + 1, 1)
            foci_layout.addWidget(y_edit, i + 1, 2)
            self._focus_edits.append((x_edit, y_edit))

        btn_layout = QHBoxLayout()
        self._btn_add = QPushButton("添加焦点")
        self._btn_add.clicked.connect(self._add_focus)
        self._btn_remove = QPushButton("删除焦点")
        self._btn_remove.clicked.connect(self._remove_focus)
        btn_layout.addWidget(self._btn_add)
        btn_layout.addWidget(self._btn_remove)
        foci_layout.addLayout(btn_layout, 4, 0, 1, 3)

        foci_group.setLayout(foci_layout)
        layout.addWidget(foci_group)

        # Canvas
        canvas_group = QGroupBox("2D 可视化 (128 x 128 mm)")
        canvas_layout = QVBoxLayout()
        self._canvas = FocusCanvas()
        canvas_layout.addWidget(self._canvas)
        canvas_group.setLayout(canvas_layout)
        layout.addWidget(canvas_group, stretch=1)

        self._update_focus_inputs()

    def _on_mode_changed(self, btn_id: int):
        self._mode = "static" if btn_id == 0 else "dynamic"
        self.mode_changed.emit(self._mode)

    def _on_foci_edited(self):
        self._sync_foci_from_ui()

    def _add_focus(self):
        if len(self._foci) >= 3:
            QMessageBox.warning(self, "已达上限", "最多允许 3 个焦点。")
            return
        # Default new focus at center
        self._foci.append((64.0, 64.0))
        self._update_focus_inputs()

    def _remove_focus(self):
        if len(self._foci) <= 1:
            QMessageBox.warning(self, "已达下限", "至少需要 1 个焦点。")
            return
        self._foci.pop()
        self._update_focus_inputs()

    def _update_focus_inputs(self):
        n = len(self._foci)
        if n == 0:
            self._foci = [(64.0, 64.0)]
            n = 1

        # Block signals to prevent recursive textChanged -> _sync_foci_from_ui
        # which could modify self._foci while we are iterating.
        for x_edit, y_edit in self._focus_edits:
            x_edit.blockSignals(True)
            y_edit.blockSignals(True)

        try:
            for i, (x_edit, y_edit) in enumerate(self._focus_edits):
                if i < n:
                    x_edit.setText(f"{self._foci[i][0]:.2f}")
                    y_edit.setText(f"{self._foci[i][1]:.2f}")
                    x_edit.setEnabled(True)
                    y_edit.setEnabled(True)
                else:
                    x_edit.clear()
                    y_edit.clear()
                    x_edit.setEnabled(False)
                    y_edit.setEnabled(False)

            self._canvas.set_foci(self._foci)
            self.foci_changed.emit(list(self._foci))
        finally:
            for x_edit, y_edit in self._focus_edits:
                x_edit.blockSignals(False)
                y_edit.blockSignals(False)

    def _sync_foci_from_ui(self):
        foci = []
        for x_edit, y_edit in self._focus_edits:
            if x_edit.isEnabled() and y_edit.isEnabled():
                x_text = x_edit.text().strip()
                y_text = y_edit.text().strip()
                if not x_text or not y_text:
                    # Skip incomplete input (user is still typing)
                    continue
                try:
                    x = float(x_text)
                    y = float(y_text)
                    foci.append((x, y))
                except ValueError:
                    pass
        if not foci:
            return  # Don't clear foci while user is editing
        self._foci = foci
        self._canvas.set_foci(foci)
        self.foci_changed.emit(list(foci))

    def get_foci(self) -> List[Tuple[float, float]]:
        return list(self._foci)

    def get_mode(self) -> str:
        return self._mode
