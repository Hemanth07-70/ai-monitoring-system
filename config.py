import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Camera settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Alert Configuration (Email)
# Note: For Gmail, use an "App Password"
ALERT_EMAIL_SENDER = os.environ.get("ALERT_EMAIL_SENDER", "")
ALERT_EMAIL_RECEIVER = os.environ.get("ALERT_EMAIL_RECEIVER", "")
ALERT_EMAIL_PASSWORD = os.environ.get("ALERT_EMAIL_PASSWORD", "")

# Verification Engine Settings
REPETITION_THRESHOLD = 3
TIME_WINDOW_SECONDS = 20
CONFIDENCE_THRESHOLD = 0.85
