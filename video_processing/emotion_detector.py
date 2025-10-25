import cv2
import numpy as np
import time
import random
from deepface import DeepFace
from config import Config

class FacialEmotionDetector:
    def __init__(self):
        self.labels = Config.EMOTION_LABELS['face']
        self.detector_backend = 'opencv'
        self.last_analysis_time = 0
        self.analysis_interval = 2  # Analyze every 2 seconds
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.emotion_history = []
        self.confidence_history = []
        self.max_history = 8
        self.face_detection_history = []
        self.session_emotions = []
        
    def detect_faces(self, frame):
        """Detect faces in the frame using OpenCV with multiple methods"""
        if frame is None:
            return []
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Primary face detection with Haar cascades
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=6,
                minSize=(50, 50),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Update face detection history
            self.face_detection_history.append(len(faces) > 0)
            if len(self.face_detection_history) > 10:
                self.face_detection_history.pop(0)
            
            return faces
            
        except Exception as e:
            print(f"Error in face detection: {e}")
            return []
    
    def preprocess_face(self, face_region):
        """Preprocess face region for better emotion detection"""
        if face_region.size == 0:
            return None
            
        try:
            # Resize to standard size for DeepFace
            target_size = (224, 224)
            resized_face = cv2.resize(face_region, target_size)
            
            # Normalize pixel values
            normalized_face = resized_face.astype(np.float32) / 255.0
            
            # Apply mild histogram equalization for better contrast
            if len(face_region.shape) == 3:
                # Convert to YUV for luminance adjustment
                yuv = cv2.cvtColor(resized_face, cv2.COLOR_RGB2YUV)
                yuv[:,:,0] = cv2.equalizeHist(yuv[:,:,0])
                enhanced_face = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
            else:
                enhanced_face = cv2.equalizeHist(resized_face)
            
            return enhanced_face
            
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            return face_region
    
    def analyze_emotion(self, frame, face_coords):
        """Analyze emotion for a specific face region using DeepFace"""
        if frame is None or len(face_coords) != 4:
            return "neutral", 0.0
            
        try:
            x, y, w, h = face_coords
            
            # Extract face region with padding
            padding = 30
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding)
            y2 = min(frame.shape[0], y + h + padding)
            
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return "neutral", 0.0
            
            # Preprocess face
            processed_face = self.preprocess_face(face_region)
            
            # Analyze using DeepFace with multiple backends
            analysis = DeepFace.analyze(
                processed_face,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            if analysis and isinstance(analysis, list) and len(analysis) > 0:
                emotion_data = analysis[0]['emotion']
                dominant_emotion = analysis[0]['dominant_emotion']
                confidence = emotion_data[dominant_emotion] / 100.0
                
                # Map to our emotion labels
                emotion_map = {
                    'angry': 'anger',
                    'disgust': 'disgust', 
                    'fear': 'fear',
                    'happy': 'happy',
                    'sad': 'sad',
                    'surprise': 'surprise',
                    'neutral': 'neutral'
                }
                
                mapped_emotion = emotion_map.get(dominant_emotion, 'neutral')
                
                # Apply confidence threshold and quality checks
                if confidence < 0.3:
                    return "neutral", 0.5
                
                # Check face quality (size, brightness, etc.)
                if not self.is_face_quality_good(face_region):
                    confidence *= 0.8  # Reduce confidence for poor quality faces
                    
                return mapped_emotion, confidence
                
        except Exception as e:
            print(f"Error in DeepFace emotion analysis: {e}")
            
        return "neutral", 0.0
    
    def is_face_quality_good(self, face_region):
        """Check if face region has good quality for emotion detection"""
        if face_region.size == 0:
            return False
            
        try:
            # Check size
            h, w = face_region.shape[:2]
            if h < 50 or w < 50:
                return False
            
            # Check brightness (convert to grayscale if needed)
            if len(face_region.shape) == 3:
                gray_face = cv2.cvtColor(face_region, cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_region
            
            brightness = np.mean(gray_face)
            if brightness < 30 or brightness > 220:  # Too dark or too bright
                return False
            
            # Check contrast
            contrast = np.std(gray_face)
            if contrast < 20:  # Low contrast
                return False
                
            return True
            
        except:
            return False
    
    def apply_temporal_smoothing(self, current_emotion, current_confidence):
        """Apply temporal smoothing to emotion detection"""
        # Add current detection to history
        self.emotion_history.append(current_emotion)
        self.confidence_history.append(current_confidence)
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        
        # If we don't have enough history, return current values
        if len(self.emotion_history) < 3:
            return current_emotion, current_confidence
        
        # Calculate emotion frequencies with confidence weighting
        emotion_scores = {}
        for i, (emotion, confidence) in enumerate(zip(self.emotion_history, self.confidence_history)):
            weight = confidence * (i + 1) / len(self.emotion_history)  # Weight recent frames more
            emotion_scores[emotion] = emotion_scores.get(emotion, 0) + weight
        
        # Find the emotion with highest weighted score
        smoothed_emotion = max(emotion_scores, key=emotion_scores.get)
        total_weight = sum(emotion_scores.values())
        smoothed_confidence = emotion_scores[smoothed_emotion] / total_weight if total_weight > 0 else current_confidence
        
        # Only use smoothed result if it's significantly better
        if smoothed_confidence > current_confidence * 0.7:
            return smoothed_emotion, min(0.95, smoothed_confidence)
        else:
            return current_emotion, current_confidence
    
    def detect_emotion(self, frame):
        """Detect emotion from frame with comprehensive processing"""
        current_time = time.time()
        
        # Throttle analysis to prevent excessive CPU usage
        if current_time - self.last_analysis_time < self.analysis_interval:
            if self.emotion_history:
                return self.emotion_history[-1], self.confidence_history[-1] if self.confidence_history else 0.5, []
            return "neutral", 0.5, []
        
        self.last_analysis_time = current_time
        
        if frame is None:
            return "neutral", 0.5, []
            
        try:
            # Detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                # No face detected
                emotion, confidence = "neutral", 0.3
                self.emotion_history.append(emotion)
                self.confidence_history.append(confidence)
                
                if len(self.emotion_history) > self.max_history:
                    self.emotion_history.pop(0)
                    self.confidence_history.pop(0)
                    
                return emotion, confidence, []
            
            # Analyze emotion for the first face (main person)
            main_face = faces[0]
            emotion, confidence = self.analyze_emotion(frame, main_face)
            
            # Apply temporal smoothing
            smoothed_emotion, smoothed_confidence = self.apply_temporal_smoothing(emotion, confidence)
            
            # Add to session emotions for statistics
            self.session_emotions.append({
                'emotion': smoothed_emotion,
                'confidence': smoothed_confidence,
                'timestamp': time.time(),
                'face_detected': True
            })
            
            # Keep session emotions manageable
            if len(self.session_emotions) > 100:
                self.session_emotions = self.session_emotions[-100:]
            
            return smoothed_emotion, smoothed_confidence, faces
            
        except Exception as e:
            print(f"Error in facial emotion detection: {e}")
            return "neutral", 0.5, []
    
    def get_emotion_statistics(self):
        """Get statistics about recent emotion detections"""
        if not self.session_emotions:
            return {}
        
        emotion_counts = {}
        total_confidence = 0
        total_detections = len(self.session_emotions)
        
        for detection in self.session_emotions:
            emotion = detection['emotion']
            confidence = detection['confidence']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            total_confidence += confidence
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_percentage = (emotion_counts[dominant_emotion] / total_detections) * 100
        average_confidence = total_confidence / total_detections
        
        # Calculate emotion transitions
        transitions = 0
        for i in range(1, len(self.session_emotions)):
            if self.session_emotions[i]['emotion'] != self.session_emotions[i-1]['emotion']:
                transitions += 1
        
        return {
            'total_detections': total_detections,
            'dominant_emotion': dominant_emotion,
            'dominant_percentage': dominant_percentage,
            'average_confidence': average_confidence,
            'emotion_distribution': emotion_counts,
            'emotion_transitions': transitions,
            'face_detection_rate': np.mean(self.face_detection_history) if self.face_detection_history else 0
        }
    
    def reset_session(self):
        """Reset session data for new recording"""
        self.session_emotions = []
        self.face_detection_history = []
        self.emotion_history = []
        self.confidence_history = []

# Enhanced Mock detector for testing
class MockFacialEmotionDetector:
    def __init__(self):
        self.labels = Config.EMOTION_LABELS['face']
        self.emotion_history = []
        self.max_history = 5
        self.session_emotions = []
        
    def detect_emotion(self, frame):
        """Mock emotion detection with temporal consistency"""
        import random
        
        # Base probabilities for emotions with more realistic distribution
        emotions_with_weights = [
            ('neutral', 0.35),
            ('happy', 0.25), 
            ('sad', 0.12),
            ('surprise', 0.10),
            ('anger', 0.08),
            ('fear', 0.06),
            ('disgust', 0.04)
        ]
        
        emotions, weights = zip(*emotions_with_weights)
        
        # Add temporal consistency
        if self.emotion_history:
            last_emotion = self.emotion_history[-1]
            # 70% chance to continue same emotion, 30% chance to change
            if random.random() < 0.7:
                emotion = last_emotion
                confidence = random.uniform(0.7, 0.9)
            else:
                emotion = random.choices(emotions, weights=weights)[0]
                confidence = random.uniform(0.6, 0.8)
        else:
            emotion = random.choices(emotions, weights=weights)[0]
            confidence = random.uniform(0.6, 0.8)
        
        # Update history
        self.emotion_history.append(emotion)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        
        # Add to session emotions
        self.session_emotions.append({
            'emotion': emotion,
            'confidence': confidence,
            'timestamp': time.time(),
            'face_detected': True
        })
        
        # Generate mock face coordinates
        if frame is not None:
            h, w = frame.shape[:2]
            face_w = random.randint(100, 200)
            face_h = random.randint(100, 200)
            face_x = random.randint(50, w - face_w - 50)
            face_y = random.randint(50, h - face_h - 50)
            faces = [(face_x, face_y, face_w, face_h)]
        else:
            faces = []
            
        return emotion, confidence, faces
    
    def get_emotion_statistics(self):
        """Get mock emotion statistics"""
        if not self.session_emotions:
            return {}
        
        emotion_counts = {}
        for detection in self.session_emotions:
            emotion = detection['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_detections = len(self.session_emotions)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_percentage = (emotion_counts[dominant_emotion] / total_detections) * 100
        
        return {
            'total_detections': total_detections,
            'dominant_emotion': dominant_emotion,
            'dominant_percentage': dominant_percentage,
            'emotion_distribution': emotion_counts,
            'average_confidence': 0.75,
            'emotion_transitions': random.randint(5, 15),
            'face_detection_rate': 0.85
        }
    
    def reset_session(self):
        """Reset mock session data"""
        self.session_emotions = []
        self.emotion_history = []