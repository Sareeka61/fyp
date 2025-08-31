import os

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER_NAME = 'anpr_uploads'
UPLOAD_FOLDER_PATH = os.path.join(APP_ROOT, UPLOAD_FOLDER_NAME)

PLATE_MODEL_PATH = os.path.join(APP_ROOT, 'models/pd_traific_v2_mix.pt')
CHAR_SEG_MODEL_PATH = os.path.join(APP_ROOT, 'models/sg_traific_v12.pt')
CHAR_REC_MODEL_PATH = os.path.join(APP_ROOT, 'models/char_traific_v3.pth')

FONT_PATH = None  

CLASS_LABELS = [
    'क', 'को', 'ख', 'ग', 'च', 'ज', 'झ', 'ञ', 'डि', 'त', 'ना', 'प', 'प्र', 'ब', 'बा',
    'भे', 'म', 'मे', 'य', 'लु', 'सी', 'सु', 'से', 'ह', '0', '१', '२', '३', '४', '५',
    '६', '७', '८', '९'
]
NUM_CLASSES = len(CLASS_LABELS)

PLATE_DETECT_CONF = 0.4
CHAR_SEG_CONF = 0.3
CHAR_REC_CONF_THRESHOLD = 0.4 

TARGET_FPS = 3  # Process 3 frames per second from videos
VIDEO_FRAME_SKIP = 10  # Default fallback value if frame rate cannot be determined

DESKEW_MIN_PLATE_HEIGHT = 15
DESKEW_MIN_PLATE_WIDTH = 30

CHAR_ORDERING_HEIGHT_FRACTION = 0.6
CHAR_ORDERING_LINE_GAP_FACTOR = 0.5

# Traffic Violation Detection Configuration
TRAFFIC_LIGHT_DETECTION_ENABLED = True
STOP_LINE_Y_COORDINATE = 1400  # Default Y-coordinate for stop line (adjust based on camera angle)
TRAFFIC_LIGHT_ROI_TOP = 0.05  # Top 5% of frame for traffic light detection
TRAFFIC_LIGHT_ROI_BOTTOM = 0.20  # Bottom of ROI at 20% of frame
TRAFFIC_LIGHT_ROI_LEFT = 0.40  # Left 40% of frame
TRAFFIC_LIGHT_ROI_RIGHT = 0.60  # Right 60% of frame
RED_LIGHT_THRESHOLD = 0.01  # 1% red pixels threshold for red light detection
GREEN_LIGHT_THRESHOLD = 0.01  # 1% green pixels threshold for green light detection
VIOLATION_CONFIDENCE_THRESHOLD = 0.3  # Minimum confidence for violation reporting

# Live Processing Configuration
TARGET_FPS = 3  # Process 3 frames per second from videos for live processing
FRAME_SIZE = (1280, 720)  # Target frame size for streaming and processing
VIDEO_FRAME_SKIP = 10  # Default fallback value if frame rate cannot be determined

# Traffic Light Simulation Configuration
RED_LIGHT_DURATION = 4.0    # Duration of red light phase in seconds
GREEN_LIGHT_DURATION = 2.0  # Duration of green light phase in seconds
TOTAL_CYCLE_TIME = RED_LIGHT_DURATION + GREEN_LIGHT_DURATION

# Virtual Line and Crossing Detection
VIRTUAL_LINE_Y_COORDINATE = 1400  # Y-coordinate for virtual stop line (adjusted for better detection)
VIRTUAL_LINE_COLOR = (0, 0, 255)  # Red color for stop line
VIRTUAL_LINE_THICKNESS = 4
VIOLATION_BUFFER_ZONE = 10  # Pixels buffer zone before stop line to prevent false positives
VIOLATION_MIN_CROSSING_DEPTH = 5  # Minimum pixels vehicle must cross beyond stop line

# Traffic Light Detection Configuration
TRAFFIC_LIGHT_ROI_X1_RATIO = 0.4  # Left boundary of traffic light ROI (as fraction of frame width)
TRAFFIC_LIGHT_ROI_Y1_RATIO = 0.05  # Top boundary of traffic light ROI (as fraction of frame height)
TRAFFIC_LIGHT_ROI_X2_RATIO = 0.6   # Right boundary of traffic light ROI (as fraction of frame width)
TRAFFIC_LIGHT_ROI_Y2_RATIO = 0.25  # Bottom boundary of traffic light ROI (as fraction of frame height)
TRAFFIC_LIGHT_COLOR_THRESHOLD = 0.5  # Minimum percentage of ROI that must be detected color (0.5%)

# Vehicle Tracking Configuration
IOU_THRESHOLD = 0.3  # Intersection over Union threshold for track matching
HYSTERESIS_FRAMES = 1  # Frames to wait before confirming violation
MAX_TRACK_AGE = 5.0  # Maximum age of track in seconds
MAX_ACTIVE_TRACKS = 20  # Maximum number of active tracks

# File Output Configuration
GENERATE_ANNOTATED_VIDEO = True
GENERATE_CSV_REPORT = True
GENERATE_JSON_REPORT = True
GENERATE_SNAPSHOTS = True
MAX_VIOLATION_SNAPSHOTS = 100  # Maximum number of violation snapshots to generate

# Streaming Configuration
STREAM_FRAME_BUFFER_SIZE = 30  # Size of frame buffer for streaming
STREAM_JPEG_QUALITY = 80  # JPEG quality for streaming (0-100)
STREAM_TARGET_WIDTH = 1280  # Target width for streaming frames
STREAM_TARGET_HEIGHT = 720  # Target height for streaming frames

# Job Management Configuration
MAX_CONCURRENT_JOBS = 100  # Maximum number of concurrent processing jobs
JOB_CLEANUP_INTERVAL = 3600  # Job cleanup interval in seconds (1 hour)
MAX_JOB_AGE_HOURS = 24  # Maximum age of completed jobs in hours

# Event Streaming Configuration
EVENT_QUEUE_SIZE = 100  # Maximum size of event queue per job
SSE_HEARTBEAT_INTERVAL = 30  # SSE heartbeat interval in seconds

FLASK_SECRET_KEY = 'your_very_secret_key_change_me'
MAX_CONTENT_LENGTH = 1000 * 1024 * 1024 
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.mp4', '.avi', '.mov', '.mkv'}
