"""
Main application window.
"""

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QStatusBar, QLabel, QMessageBox,
)
from PySide6.QtCore import Qt

# Ensure host/ is on path when run directly
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from algorithms.engine import UNetEngine, GSPATEngine
from algorithms.field_builder_torch import FieldBuilderTorch
from core.converter import ControlMatrixConverter
from core.device_client import DeviceClient
from core.renderer import RenderController
from gui.focus_panel import FocusPanel
from gui.param_panel import ParamPanel
from gui.visualize_widget import FieldVisualizationWidget

import config as cfg


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("超声换能器阵列控制器")
        self.setMinimumSize(1200, 800)

        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # Left: Focus panel
        left_layout = QVBoxLayout()
        self.focus_panel = FocusPanel()
        self.focus_panel.foci_changed.connect(self._on_config_changed)
        self.focus_panel.mode_changed.connect(self._on_config_changed)
        left_layout.addWidget(self.focus_panel)

        # Right: Parameters + Field visualization
        right_layout = QVBoxLayout()
        self.param_panel = ParamPanel()
        self.param_panel.algorithm_changed.connect(self._on_config_changed)
        self.param_panel.lm_params_changed.connect(self._on_config_changed)
        self.param_panel.dwell_time_changed.connect(self._on_config_changed)
        self.param_panel.connect_requested.connect(self._on_connect)
        self.param_panel.disconnect_requested.connect(self._on_disconnect)
        self.param_panel.start_requested.connect(self._on_start)
        self.param_panel.stop_requested.connect(self._on_stop)
        right_layout.addWidget(self.param_panel)

        self.viz_widget = FieldVisualizationWidget()
        right_layout.addWidget(self.viz_widget, stretch=1)

        main_layout.addLayout(left_layout, stretch=1)
        main_layout.addLayout(right_layout, stretch=2)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self._status_connection = QLabel("设备: 未连接")
        self._status_fps = QLabel("FPS: --")
        self._status_algo = QLabel("算法: U-Net")
        self._status_mode = QLabel("模式: 静态")
        self.status_bar.addWidget(self._status_connection)
        self.status_bar.addWidget(self._status_fps)
        self.status_bar.addWidget(self._status_algo)
        self.status_bar.addWidget(self._status_mode)

        # Initialize engines
        self._init_engines()

        # Initial configuration
        self._configure_renderer()

    def _init_engines(self):
        """Initialize U-Net and GS-PAT engines."""
        try:
            checkpoint_path = cfg.CHECKPOINT_PATH
            if not checkpoint_path.exists():
                QMessageBox.warning(
                    self, "未找到模型检查点",
                    f"U-Net 检查点未找到: {checkpoint_path}\n"
                    "U-Net 算法将不可用。"
                )
                self.unet_engine = None
            else:
                self.unet_engine = UNetEngine(
                    checkpoint_path=str(checkpoint_path),
                    base_channels=cfg.UNET_BASE_CHANNELS,
                    use_compile=False,  # Avoid compile overhead for GUI responsiveness
                )
        except Exception as e:
            QMessageBox.warning(self, "引擎错误", f"加载 U-Net 引擎失败:\n{e}")
            self.unet_engine = None

        self.gs_pat_engine = GSPATEngine(
            f=cfg.F,
            width=cfg.ARRAY_WIDTH,
            height=cfg.ARRAY_HEIGHT,
            d=cfg.D,
            z=cfg.Z,
            c=cfg.c,
            max_iter=cfg.GS_PAT_MAX_ITER,
        )

        # Field builder for visualization
        self.field_builder = FieldBuilderTorch(
            f=cfg.F,
            array_width=cfg.ARRAY_WIDTH,
            array_height=cfg.ARRAY_HEIGHT,
            d=cfg.D,
            z=cfg.Z,
            image_resolution=cfg.IMAGE_RESOLUTION,
            image_size=cfg.IMAGE_SIZE,
            c=cfg.c,
        )

        # Device client (mock initially)
        self.device_client = DeviceClient(
            host=cfg.DEFAULT_DEVICE_IP,
            port=cfg.DEFAULT_DEVICE_PORT,
            mock=True,
            on_status_change=self._on_device_status,
        )

        # Render controller
        self.renderer = RenderController(
            unet_engine=self.unet_engine,
            gs_pat_engine=self.gs_pat_engine,
            device_client=self.device_client,
            field_builder=self.field_builder,
        )
        self.renderer.field_visualization.connect(self.viz_widget.update_field)
        self.renderer.status_message.connect(self._on_renderer_status)
        self.renderer.fps_updated.connect(self._on_fps_updated)
        self.renderer.connection_status.connect(self._on_renderer_conn_status)
        self.renderer.current_focus_info.connect(self._on_focus_info)

    def _configure_renderer(self):
        """Apply current GUI configuration to the renderer."""
        try:
            foci = self.focus_panel.get_foci()
            mode = self.focus_panel.get_mode()
            algo = self.param_panel.get_algorithm()
            lm = self.param_panel.get_lm_params()
            dwell = self.param_panel.get_dwell_time_ms()

            # If U-Net is not available, force GS-PAT
            if algo == "unet" and self.unet_engine is None:
                algo = "gs_pat"
                self.param_panel._algo_combo.setCurrentIndex(1)

            self.renderer.configure(
                foci=foci,
                mode=mode,
                algorithm=algo,
                lm_freq=lm["freq"],
                lm_amp=lm["amp"],
                lm_samples=lm["samples"],
                lm_direction=lm["direction"],
                dwell_time_ms=dwell,
            )

            self._status_algo.setText(f"算法: {'U-Net' if algo == 'unet' else 'GS-PAT'}")
            self._status_mode.setText(f"模式: {'静态' if mode == 'static' else '动态'}")
        except Exception as e:
            self.status_bar.showMessage(f"配置错误: {e}", 5000)

    def _on_config_changed(self):
        """Called when any configuration parameter changes."""
        if self.renderer.is_running():
            # For simplicity, stop and reconfigure
            self.renderer.stop()
            self.param_panel._btn_start.setEnabled(True)
            self.param_panel._btn_stop.setEnabled(False)
        self._configure_renderer()

    def _on_connect(self, ip: str, port: int):
        """Handle connect button."""
        if self.device_client.connected:
            self._release_connection()

        mock = self.param_panel.is_mock()
        self.device_client.mock = mock
        self.device_client.host = ip
        self.device_client.port = port

        ok = self.device_client.connect()
        if ok:
            self.param_panel.set_connected(True)
            self._status_connection.setText(f"设备: {'模拟' if mock else '已连接'}")
        else:
            QMessageBox.critical(self, "连接失败", f"无法连接到 {ip}:{port}")

    def _on_disconnect(self):
        """Handle disconnect button (e.g. after PS ARM reflashed)."""
        self._release_connection()
        self.status_bar.showMessage("已断开设备连接", 3000)

    def _release_connection(self):
        """Stop streaming and close TCP; safe to call when already disconnected."""
        if self.renderer.is_running():
            self.renderer.stop()
        self.device_client.disconnect()
        self.param_panel.set_connected(False)
        self.param_panel._btn_start.setEnabled(False)
        self.param_panel._btn_stop.setEnabled(False)
        self._status_connection.setText("设备: 未连接")

    def _on_start(self):
        """Handle start rendering button."""
        if not self.device_client.connected:
            QMessageBox.warning(self, "未连接", "请先连接设备。")
            self.param_panel._btn_start.setEnabled(False)
            return
        self.renderer.start()

    def _on_stop(self):
        """Handle stop rendering button."""
        self.renderer.stop()

    def _on_device_status(self, status: str):
        """Callback from DeviceClient."""
        if status in ("DISCONNECTED", "MOCK_DISCONNECTED"):
            self.param_panel.set_connected(False)
            self.param_panel._btn_start.setEnabled(False)
            self.param_panel._btn_stop.setEnabled(False)
            self._status_connection.setText("设备: 未连接")
            return
        if status.startswith("SEND_ERROR") or status.startswith("ERROR"):
            self._release_connection()
            self.status_bar.showMessage(f"连接异常: {status}", 5000)
            return
        self._status_connection.setText(f"设备: {status}")

    def _on_renderer_status(self, msg: str):
        self.status_bar.showMessage(msg, 3000)

    def _on_fps_updated(self, fps: float):
        self._status_fps.setText(f"FPS: {fps:.1f}")

    def _on_renderer_conn_status(self, status: str):
        if status == "发送失败":
            self._release_connection()
            self.status_bar.showMessage("发送失败，已断开连接", 5000)
        else:
            self._status_connection.setText(f"设备: {status}")

    def _on_focus_info(self, info: str):
        self._status_mode.setText(f"模式: {info}")

    def closeEvent(self, event):
        """Clean up on window close."""
        self.renderer.stop()
        self.device_client.disconnect()
        event.accept()
