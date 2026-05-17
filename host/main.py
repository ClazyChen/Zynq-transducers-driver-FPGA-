"""
Entry point for the host PC application.
"""

import sys
from pathlib import Path

# Ensure host/ is on path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Ultrasound Transducer Array Controller")
    app.setOrganizationName("ZynqFPGA")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
