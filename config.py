import os

# Model and paths
MODEL_PATH = "my_keras_model"
LOG_FILE = "exercise_log.csv"

# Pose classes
POSE_CLASSES = ["sqs", "shf", "ars"]

# Thresholds
CORRECT_THRESHOLD = 0.65
RETRY_THRESHOLD = 0.45
EVAL_DELAY = 0.15
REST_TIME = 5
VERIFICATION_COUNT = 3

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# Window settings
WINDOW_NAME = "PE Guide"
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Speech settings
SPEECH_RATE = 165

# Tones
TONE_SUCCESS_FREQ = 880
TONE_SUCCESS_DURATION = 0.3
TONE_ALMOST_FREQ = 660
TONE_ALMOST_DURATION = 0.25
TONE_FAIL_FREQ = 220
TONE_FAIL_DURATION = 0.4
