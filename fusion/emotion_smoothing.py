import time
from collections import deque

class EmotionSmoothing:
    def __init__(self, history_size=5):
        """
        Initializes the EmotionSmoothing object.
        
        Args:
            history_size (int): Number of frames to average over (default: 5).
        """
        self.emotion_history = deque(maxlen=history_size)
    
    def smooth_prediction(self, emotion, confidence):
        """
        Averages the predictions across the last N frames to provide a stable emotion.
        
        Args:
            emotion (str): The current frame's detected emotion.
            confidence (float): The current frame's confidence score.
            
        Returns:
            dict: The smoothed emotion and averaged confidence.
        """
        if emotion is None or confidence is None:
            return self._get_current_stable()
            
        self.emotion_history.append({
            "emotion": emotion,
            "confidence": confidence
        })
        
        return self._get_current_stable()
        
    def _get_current_stable(self):
        """
        Computes the most frequent emotion and average confidence in the history window.
        """
        if not self.emotion_history:
            return {"emotion": "neutral", "confidence": 0.0}
            
        # Count frequencies of each emotion in the queue
        emotion_counts = {}
        confidence_sums = {}
        
        for record in self.emotion_history:
            em = record["emotion"]
            conf = record["confidence"]
            
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
            confidence_sums[em] = confidence_sums.get(em, 0.0) + conf
            
        # Find the most frequent emotion (the stable one)
        stable_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Calculate its average confidence
        avg_confidence = confidence_sums[stable_emotion] / emotion_counts[stable_emotion]
        
        return {
            "emotion": stable_emotion,
            "confidence": avg_confidence
        }
