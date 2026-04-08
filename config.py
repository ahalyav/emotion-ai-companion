import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = True

    # Audio
    AUDIO_SAMPLE_RATE = 44100
    AUDIO_CHUNK_SIZE = 1024
    AUDIO_FORMAT = 'float32'
    AUDIO_CHANNELS = 1
    AUDIO_SMOOTHING_ALPHA = 0.3      # EMA coefficient for temporal smoothing

    # Video
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    VIDEO_FPS = 30
    FACE_ANALYSIS_INTERVAL = 0.5    # seconds between face analysis

    # Models
    AUDIO_MODEL_NAME = "superb/wav2vec2-base-superb-erm"
    FACE_MODEL_NAME = "deepface"

    # Gemini API
    GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', '')
    GEMINI_MODEL = 'gemini-1.5-flash'

    # Ollama (optional fallback)
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"

    # Emotion Labels
    EMOTION_LABELS = {
        'audio': ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'face': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'combined': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
    }

    # Fusion Weights (audio slightly dominant as it's richer signal)
    FUSION_WEIGHTS = {
        'audio': 0.55,
        'face': 0.45,
    }

    # History / Recording
    EMOTION_HISTORY_DEPTH = 60       # number of timestamped snapshots to keep
    MAX_RECORDING_DURATION = 300     # 5 minutes
    SESSION_SAVE_PATH = "recorded_sessions"
    MAX_SESSIONS = 50

    # Chat
    CHAT_HISTORY_MAX = 10            # rolling chat message pairs kept in context