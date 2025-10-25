import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Flask Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')
    DEBUG = True
    
    # Audio Configuration - UPDATED to match your system
    AUDIO_SAMPLE_RATE = 44100  # Changed from 16000 to 44100
    AUDIO_CHUNK_SIZE = 1024
    AUDIO_FORMAT = 'float32'
    AUDIO_CHANNELS = 1
    
    # Video Configuration
    CAMERA_INDEX = 0
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    VIDEO_FPS = 30
    
    # Model Configuration
    AUDIO_MODEL_NAME = "superb/wav2vec2-base-superb-erm"
    FACE_MODEL_NAME = "deepface"
    
    # Ollama Configuration
    OLLAMA_BASE_URL = "http://localhost:11434"
    OLLAMA_MODEL = "llama2"
    
    # Emotion Labels
    EMOTION_LABELS = {
        'audio': ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'face': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral'],
        'combined': ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral', 'confused']
    }
    
    # Fusion Weights
    FUSION_WEIGHTS = {
        'audio': 0.6,
        'face': 0.4
    }
    
    # Recording Configuration
    MAX_RECORDING_DURATION = 300  # 5 minutes in seconds
    SESSION_SAVE_PATH = "recorded_sessions"
    MAX_SESSIONS = 50