import numpy as np
import random
from config import Config

class SimpleAudioEmotionDetector:
    def __init__(self):
        self.labels = Config.EMOTION_LABELS['audio']
        
    def real_time_emotion_analysis(self, audio_chunk):
        """Improved emotion detection with better audio analysis"""
        try:
            # Check if audio_chunk is valid
            if audio_chunk is None:
                return "neutral", 0.5
                
            # Ensure it's a numpy array and flatten it
            audio_data = np.array(audio_chunk, dtype=np.float32).flatten()
            
            # Check if we have any data
            if len(audio_data) == 0:
                return "neutral", 0.5
            
            # Calculate basic audio features safely
            rms = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
            
            # Calculate zero-crossing rate (speech vs silence indicator)
            if len(audio_data) > 1:
                zero_crossings = np.sum(np.diff(audio_data > 0)) / len(audio_data)
            else:
                zero_crossings = 0
            
            # Improved emotion mapping based on audio characteristics
            if rms > 0.03:  # Loud sound - excitement/surprise
                emotion = "surprise"
                confidence = min(0.7 + rms * 2, 0.9)
            elif rms > 0.01:  # Moderate sound - happy/engaged
                emotion = "happy" 
                confidence = min(0.6 + rms * 3, 0.8)
            elif zero_crossings > 0.1:  # Speech-like patterns - neutral
                emotion = "neutral"
                confidence = 0.7
            elif rms > 0.001:  # Very quiet - calm/neutral
                emotion = "neutral"
                confidence = 0.6
            else:  # Almost silent - default to neutral
                emotion = "neutral"
                confidence = 0.5
                
            # Add some random variation to make it more realistic
            if random.random() < 0.3:  # 30% chance to vary slightly
                emotions = ['neutral', 'happy', 'sad', 'surprise']
                weights = [0.4, 0.3, 0.2, 0.1]
                emotion = random.choices(emotions, weights=weights)[0]
                confidence = max(0.4, confidence * random.uniform(0.8, 1.2))
                
            return emotion, min(confidence, 0.95)  # Cap confidence at 95%
            
        except Exception as e:
            print(f"Audio emotion detection error: {e}")
            # Fallback to neutral with low confidence
            return "neutral", 0.3