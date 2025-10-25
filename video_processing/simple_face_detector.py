import cv2
import numpy as np
import time
import random
from config import Config

class SimpleFaceDetector:
    def __init__(self):
        self.labels = Config.EMOTION_LABELS['face']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.last_analysis_time = 0
        self.analysis_interval = 2
        self.emotion_history = []
        self.max_history = 5
        
    def detect_faces(self, frame):
        """Simple face detection using OpenCV"""
        if frame is None:
            return []
            
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50)
            )
            return faces
        except Exception as e:
            print(f"Face detection error: {e}")
            return []
    
    def estimate_emotion_from_face(self, face_region):
        """Simple emotion estimation based on facial features"""
        if face_region.size == 0:
            return "neutral", 0.5
            
        try:
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            
            # Basic feature analysis
            brightness = np.mean(gray_face)
            contrast = np.std(gray_face)
            
            # Simple emotion mapping
            if brightness > 160 and contrast > 45:
                return "happy", 0.8
            elif brightness < 80 and contrast < 25:
                return "sad", 0.7
            elif contrast > 55:
                return "surprise", 0.75
            elif brightness > 170:
                return "anger", 0.7
            else:
                return "neutral", 0.6
                
        except Exception as e:
            print(f"Emotion estimation error: {e}")
            return "neutral", 0.5
    
    def smooth_emotion(self, current_emotion, current_confidence):
        """Simple temporal smoothing"""
        self.emotion_history.append((current_emotion, current_confidence))
        
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        
        if len(self.emotion_history) < 2:
            return current_emotion, current_confidence
        
        emotion_counts = {}
        for emotion, confidence in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        most_common_emotion = max(emotion_counts, key=emotion_counts.get)
        frequency = emotion_counts[most_common_emotion] / len(self.emotion_history)
        
        if frequency > 0.6 and most_common_emotion != current_emotion:
            return most_common_emotion, min(0.9, current_confidence * 1.1)
        else:
            return current_emotion, current_confidence
    
    def detect_emotion(self, frame):
        """Main emotion detection method"""
        current_time = time.time()
        
        # Throttle analysis
        if current_time - self.last_analysis_time < self.analysis_interval:
            if self.emotion_history:
                last_emotion, last_confidence = self.emotion_history[-1]
                return last_emotion, last_confidence, []
            return "neutral", 0.5, []
        
        self.last_analysis_time = current_time
        
        if frame is None:
            return "neutral", 0.5, []
            
        try:
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                emotion, confidence = "neutral", 0.3
                self.emotion_history.append((emotion, confidence))
                return emotion, confidence, []
            
            x, y, w, h = faces[0]
            face_region = frame[y:y+h, x:x+w]
            
            emotion, confidence = self.estimate_emotion_from_face(face_region)
            smoothed_emotion, smoothed_confidence = self.smooth_emotion(emotion, confidence)
            
            return smoothed_emotion, smoothed_confidence, faces
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "neutral", 0.5, []