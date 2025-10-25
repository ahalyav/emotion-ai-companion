import numpy as np
from config import Config

class EmotionFusion:
    def __init__(self):
        self.audio_weight = Config.FUSION_WEIGHTS['audio']
        self.face_weight = Config.FUSION_WEIGHTS['face']
        self.emotion_labels = Config.EMOTION_LABELS['combined']
        
        # Emotion mapping between audio and face models
        self.emotion_mapping = {
            'audio': {
                'anger': 'angry',
                'disgust': 'disgust', 
                'fear': 'fear',
                'happy': 'happy',
                'sad': 'sad', 
                'surprise': 'surprise',
                'neutral': 'neutral'
            },
            'face': {
                'angry': 'angry',
                'disgust': 'disgust',
                'fear': 'fear', 
                'happy': 'happy',
                'sad': 'sad',
                'surprise': 'surprise',
                'neutral': 'neutral'
            }
        }
    
    def fuse_emotions(self, audio_emotion, audio_confidence, face_emotion, face_confidence):
        """Fuse audio and facial emotion predictions"""
        
        # If one modality is unavailable, return the other
        if audio_emotion == 'unavailable' and face_emotion == 'unavailable':
            return 'neutral', 0.0
            
        if audio_emotion == 'unavailable':
            return face_emotion, face_confidence
            
        if face_emotion == 'unavailable':
            return audio_emotion, audio_confidence
        
        # Map emotions to common labels
        audio_mapped = self.emotion_mapping['audio'].get(audio_emotion, 'neutral')
        face_mapped = self.emotion_mapping['face'].get(face_emotion, 'neutral')
        
        # If emotions match, use weighted average
        if audio_mapped == face_mapped:
            fused_confidence = (audio_confidence * self.audio_weight + 
                              face_confidence * self.face_weight)
            return audio_mapped, fused_confidence
        
        # If emotions don't match, choose the one with higher confidence
        audio_score = audio_confidence * self.audio_weight
        face_score = face_confidence * self.face_weight
        
        if audio_score > face_score:
            return audio_mapped, audio_score
        else:
            return face_mapped, face_score
    
    def get_fusion_analysis(self, audio_data, face_data):
        """Get comprehensive fusion analysis"""
        audio_emotion = audio_data.get('emotion', 'neutral')
        audio_confidence = audio_data.get('confidence', 0.0)
        face_emotion = face_data.get('emotion', 'neutral') 
        face_confidence = face_data.get('confidence', 0.0)
        
        fused_emotion, fused_confidence = self.fuse_emotions(
            audio_emotion, audio_confidence, face_emotion, face_confidence
        )
        
        return {
            'fused_emotion': fused_emotion,
            'fused_confidence': fused_confidence,
            'audio_emotion': audio_emotion,
            'audio_confidence': audio_confidence,
            'face_emotion': face_emotion,
            'face_confidence': face_confidence,
            'modality_agreement': audio_emotion == face_emotion,
            'fusion_method': 'weighted_average' if audio_emotion == face_emotion else 'confidence_based'
        }

    def get_session_summary(self, session_data):
        """Generate emotion summary for a recording session"""
        if not session_data:
            return {}
        
        emotions = session_data.get('emotions', [])
        if not emotions:
            return {}
        
        # Calculate emotion statistics
        emotion_counts = {}
        total_emotions = len(emotions)
        
        for emotion_data in emotions:
            emotion = emotion_data.get('emotion', 'neutral')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Find dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_percentage = (emotion_counts[dominant_emotion] / total_emotions) * 100
        
        return {
            'total_frames': total_emotions,
            'dominant_emotion': dominant_emotion,
            'dominant_percentage': dominant_percentage,
            'emotion_distribution': emotion_counts,
            'session_duration': session_data.get('duration', 0)
        }