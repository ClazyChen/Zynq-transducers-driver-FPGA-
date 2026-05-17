"""
Parameter configuration panel.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit, QGroupBox,
    QButtonGroup, QRadioButton, QGridLayout,
)
from PySide6.QtCore import Signal


class ParamPanel(QWidget):
    """Panel for algorithm, LM, network, and control parameters."""

    algorithm_changed = Signal(str)           # "unet" or "gs_pat"
    lm_params_changed = Signal(dict)          # {freq, amp, samples, direction}
    dwell_time_changed = Signal(int)          # ms
    connect_requested = Signal(str, int)      # ip, port
    start_requested = Signal()
    stop_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)

        # Algorithm selection
        algo_group = QGroupBox("算法")
        algo_layout = QHBoxLayout()
        self._algo_combo = QComboBox()
        self._algo_combo.addItem("U-Net (32b) - 质量优先", "unet")
        self._algo_combo.addItem("GS-PAT - 速度优先", "gs_pat")
        self._algo_combo.currentIndexChanged.connect(self._on_algo_changed)
        algo_layout.addWidget(self._algo_combo)
        algo_group.setLayout(algo_layout)
        layout.addWidget(algo_group)

        # LM parameters
        lm_group = QGroupBox("Lateral Modulation（横向调制）")
        lm_layout = QGridLayout()

        lm_layout.addWidget(QLabel("调制频率 (Hz):"), 0, 0)
        self._lm_freq = QDoubleSpinBox()
        self._lm_freq.setRange(1.0, 200.0)
        self._lm_freq.setValue(25.0)
        self._lm_freq.setDecimals(1)
        self._lm_freq.valueChanged.connect(self._on_lm_changed)
        lm_layout.addWidget(self._lm_freq, 0, 1)

        lm_layout.addWidget(QLabel("调制幅度 (mm):"), 1, 0)
        self._lm_amp = QDoubleSpinBox()
        self._lm_amp.setRange(0.1, 20.0)
        self._lm_amp.setValue(4.0)
        self._lm_amp.setDecimals(1)
        self._lm_amp.valueChanged.connect(self._on_lm_changed)
        lm_layout.addWidget(self._lm_amp, 1, 1)

        lm_layout.addWidget(QLabel("每周期采样点数:"), 2, 0)
        self._lm_samples = QSpinBox()
        self._lm_samples.setRange(2, 40)
        self._lm_samples.setValue(12)
        self._lm_samples.valueChanged.connect(self._on_lm_changed)
        lm_layout.addWidget(self._lm_samples, 2, 1)

        lm_layout.addWidget(QLabel("调制方向:"), 3, 0)
        dir_layout = QHBoxLayout()
        self._dir_group = QButtonGroup(self)
        self._rb_x = QRadioButton("X")
        self._rb_y = QRadioButton("Y")
        self._rb_x.setChecked(True)
        self._dir_group.addButton(self._rb_x)
        self._dir_group.addButton(self._rb_y)
        self._dir_group.buttonClicked.connect(self._on_lm_changed)
        dir_layout.addWidget(self._rb_x)
        dir_layout.addWidget(self._rb_y)
        lm_layout.addLayout(dir_layout, 3, 1)

        lm_group.setLayout(lm_layout)
        layout.addWidget(lm_group)

        # Dynamic mode parameters
        dyn_group = QGroupBox("动态模式")
        dyn_layout = QHBoxLayout()
        dyn_layout.addWidget(QLabel("停留时间 (ms):"))
        self._dwell_time = QSpinBox()
        self._dwell_time.setRange(50, 10000)
        self._dwell_time.setValue(500)
        self._dwell_time.setSingleStep(100)
        self._dwell_time.valueChanged.connect(self._on_dwell_changed)
        dyn_layout.addWidget(self._dwell_time)
        dyn_group.setLayout(dyn_layout)
        layout.addWidget(dyn_group)

        # Network configuration
        net_group = QGroupBox("设备网络")
        net_layout = QGridLayout()
        net_layout.addWidget(QLabel("IP 地址:"), 0, 0)
        self._ip_edit = QLineEdit("192.168.1.10")
        net_layout.addWidget(self._ip_edit, 0, 1)
        net_layout.addWidget(QLabel("端口:"), 1, 0)
        self._port_edit = QSpinBox()
        self._port_edit.setRange(1, 65535)
        self._port_edit.setValue(5000)
        net_layout.addWidget(self._port_edit, 1, 1)

        self._btn_connect = QPushButton("连接")
        self._btn_connect.clicked.connect(self._on_connect)
        net_layout.addWidget(self._btn_connect, 2, 0, 1, 2)

        net_group.setLayout(net_layout)
        layout.addWidget(net_group)

        # Control buttons
        ctrl_group = QGroupBox("控制")
        ctrl_layout = QHBoxLayout()
        self._btn_start = QPushButton("开始渲染")
        self._btn_start.setEnabled(False)
        self._btn_start.clicked.connect(self._on_start)
        self._btn_stop = QPushButton("停止渲染")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop)
        ctrl_layout.addWidget(self._btn_start)
        ctrl_layout.addWidget(self._btn_stop)
        ctrl_group.setLayout(ctrl_layout)
        layout.addWidget(ctrl_group)

        # Mock mode checkbox for debugging
        self._btn_mock = QPushButton("启用模拟模式（调试用）")
        self._btn_mock.setCheckable(True)
        self._btn_mock.setChecked(True)
        layout.addWidget(self._btn_mock)

        layout.addStretch(1)

    def _on_algo_changed(self):
        algo = self._algo_combo.currentData()
        self.algorithm_changed.emit(algo)

    def _on_lm_changed(self):
        params = {
            "freq": self._lm_freq.value(),
            "amp": self._lm_amp.value(),
            "samples": self._lm_samples.value(),
            "direction": "x" if self._rb_x.isChecked() else "y",
        }
        self.lm_params_changed.emit(params)

    def _on_dwell_changed(self):
        self.dwell_time_changed.emit(self._dwell_time.value())

    def _on_connect(self):
        ip = self._ip_edit.text().strip()
        port = self._port_edit.value()
        self.connect_requested.emit(ip, port)

    def _on_start(self):
        self.start_requested.emit()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)

    def _on_stop(self):
        self.stop_requested.emit()
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

    def set_connected(self, connected: bool):
        self._btn_connect.setEnabled(not connected)
        self._btn_start.setEnabled(connected)
        if not connected:
            self._btn_start.setEnabled(False)
            self._btn_stop.setEnabled(False)

    def is_mock(self) -> bool:
        return self._btn_mock.isChecked()

    def get_algorithm(self) -> str:
        return self._algo_combo.currentData()

    def get_lm_params(self) -> dict:
        return {
            "freq": self._lm_freq.value(),
            "amp": self._lm_amp.value(),
            "samples": self._lm_samples.value(),
            "direction": "x" if self._rb_x.isChecked() else "y",
        }

    def get_dwell_time_ms(self) -> int:
        return self._dwell_time.value()

    def get_network_config(self) -> tuple:
        return self._ip_edit.text().strip(), self._port_edit.value()
