"""
Global configuration constants for the host PC application.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.resolve()
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "checkpoint_best.pth"

# Physical constants (must match FPGA / legacy parameters)
F = 40e3                    # Frequency (Hz)
ARRAY_WIDTH = 8             # Transducer array width
ARRAY_HEIGHT = 8            # Transducer array height
D = 10.0                    # Element spacing (mm)
Z = 50.0                    # Field plane distance from array (mm)
IMAGE_SIZE = 128.0          # Image physical size (mm)
IMAGE_RESOLUTION = 128      # Image pixel resolution
c = 343000.0                # Speed of sound (mm/s)

# FPGA parameters
DIV = 30                    # Number of frames per 40kHz cycle
PHASE_DUTY_BITS = 5         # log2Ceil(DIV)
BRAM_DEPTH = 262144         # BRAM words (4096 frame slots @ 64 words/frame)

# Device streaming (40 kHz burst send to match FPGA consumption)
DEVICE_SAMPLE_RATE = 40000.0
BURST_NOMINAL_FRAMES = 512
BURST_NOMINAL_INTERVAL = BURST_NOMINAL_FRAMES / DEVICE_SAMPLE_RATE  # 0.0128 s
BURST_MIN_FRAMES = 1
# First burst after Start uses max(delta_frames, BURST_NOMINAL_FRAMES * multiplier) to prime BRAM
BURST_PRIME_MULTIPLIER = 2

# LM (Lateral Modulation) defaults
LM_DEFAULT_FREQUENCY = 25.0     # Hz
LM_DEFAULT_AMPLITUDE = 4.0      # mm
LM_DEFAULT_SAMPLES_PER_PERIOD = 12
LM_DEFAULT_DIRECTION = "x"      # "x" or "y"

# Rendering limits
MAX_FOCI = 3
MAX_SAMPLE_RATE = 40000.0       # Hz, transducer native frequency

# Network defaults (align with PS lwIP server; see send_numbers.py)
DEFAULT_DEVICE_IP = "192.168.1.20"
DEFAULT_DEVICE_PORT = 55555

# U-Net model parameters
UNET_BASE_CHANNELS = 32         # User explicitly says U-Net(32b)
UNET_IN_CHANNELS = 1
UNET_OUT_CHANNELS = 1
UNET_OUTPUT_SIZE = (8, 8)

# GS-PAT parameters
GS_PAT_MAX_ITER = 100
GS_PAT_COMPLICATED_CONTROL = False

# GUI refresh rate for field visualization (Hz)
FIELD_VISUALIZATION_FPS = 30
