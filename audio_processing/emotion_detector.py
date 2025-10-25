import numpy as np
import librosa
import torch
from config import Config

class MFCCEmotionDetector:
    """Advanced emotion detector using MFCC features and spectral analysis"""
    
    def __init__(self):
        self.labels = Config.EMOTION_LABELS['audio']
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.mfcc_features = 13
        self.feature_history = []
        self.max_history = 10
        self.emotion_history = []
        self.confidence_history = []
        
    def extract_mfcc_features(self, audio_data):
        """Extract comprehensive MFCC features from audio"""
        if audio_data is None or len(audio_data) < 512:
            return None
            
        try:
            # Ensure proper audio length
            if len(audio_data) < 2048:
                # Pad if too short
                audio_data = np.pad(audio_data, (0, 2048 - len(audio_data)), mode='constant')
            else:
                # Take first 2 seconds if too long
                max_samples = 2 * self.sample_rate
                audio_data = audio_data[:max_samples]
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=self.mfcc_features,
                n_fft=1024,
                hop_length=512
            )
            
            # Extract additional features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)
            rms_energy = librosa.feature.rms(y=audio_data)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)
            chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
            
            # Calculate statistics for each feature
            features = {}
            
            # MFCC statistics
            features['mfcc_mean'] = np.mean(mfccs, axis=1)
            features['mfcc_std'] = np.std(mfccs, axis=1)
            features['mfcc_delta'] = np.mean(librosa.feature.delta(mfccs), axis=1)
            features['mfcc_delta2'] = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
            
            # Spectral features
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            features['zero_crossing_mean'] = np.mean(zero_crossing_rate)
            features['rms_mean'] = np.mean(rms_energy)
            features['rms_std'] = np.std(rms_energy)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            
            # Temporal features
            features['pitch'] = self.estimate_pitch(audio_data)
            features['harmonics'] = self.detect_harmonics(audio_data)
            features['tempo'] = self.estimate_tempo(audio_data)
            
            return features
            
        except Exception as e:
            print(f"Error extracting MFCC features: {e}")
            return None
    
    def estimate_pitch(self, audio_data):
        """Estimate pitch using autocorrelation"""
        if len(audio_data) < 1024:
            return 0
            
        try:
            # Simple autocorrelation for pitch detection
            correlation = np.correlate(audio_data, audio_data, mode='full')
            correlation = correlation[len(correlation)//2:]
            
            # Find the first peak after the zero-lag peak
            min_lag = int(self.sample_rate / 400)  # 400Hz max
            max_lag = int(self.sample_rate / 80)   # 80Hz min
            
            if len(correlation) > max_lag:
                segment = correlation[min_lag:max_lag]
                if len(segment) > 0:
                    peak_idx = np.argmax(segment) + min_lag
                    if peak_idx > 0:
                        return self.sample_rate / peak_idx
            return 0
        except:
            return 0
    
    def detect_harmonics(self, audio_data):
        """Detect harmonic content in audio"""
        if len(audio_data) < 2048:
            return 0
            
        try:
            # Compute FFT
            fft = np.fft.fft(audio_data)
            frequencies = np.fft.fftfreq(len(fft), 1/self.sample_rate)
            
            # Get magnitude spectrum
            magnitude = np.abs(fft)
            positive_freq_idx = frequencies > 0
            magnitude = magnitude[positive_freq_idx]
            frequencies = frequencies[positive_freq_idx]
            
            # Find fundamental frequency (approximate)
            if len(magnitude) > 0:
                fundamental_idx = np.argmax(magnitude[:len(magnitude)//2])
                fundamental_freq = frequencies[fundamental_idx]
                
                # Check for harmonics
                harmonic_count = 0
                for i in range(2, 6):  # Check up to 5th harmonic
                    harmonic_freq = fundamental_freq * i
                    if harmonic_freq < frequencies[-1]:
                        # Find peak near harmonic frequency
                        harmonic_region = (frequencies > harmonic_freq * 0.9) & (frequencies < harmonic_freq * 1.1)
                        if np.any(harmonic_region) and np.max(magnitude[harmonic_region]) > np.max(magnitude) * 0.1:
                            harmonic_count += 1
                
                return harmonic_count
            return 0
        except:
            return 0
    
    def estimate_tempo(self, audio_data):
        """Estimate tempo/rhythm features"""
        if len(audio_data) < 4096:
            return 0
            
        try:
            # Use onset detection for tempo estimation
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, 
                sr=self.sample_rate, 
                hop_length=512,
                backtrack=False
            )
            
            if len(onset_frames) > 1:
                # Calculate average time between onsets
                onset_times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=512)
                intervals = np.diff(onset_times)
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    return 60 / avg_interval if avg_interval > 0 else 0  # Convert to BPM
            return 0
        except:
            return 0
    
    def analyze_emotion_from_features(self, features):
        """Analyze emotion based on comprehensive audio features"""
        if features is None:
            return "neutral", 0.5
        
        # Get feature values
        mfcc_mean = features['mfcc_mean']
        spectral_centroid = features['spectral_centroid_mean']
        zero_crossing = features['zero_crossing_mean']
        rms_energy = features['rms_mean']
        pitch = features['pitch']
        harmonics = features['harmonics']
        tempo = features['tempo']
        
        # Emotion decision tree based on research
        emotion_scores = {
            'happy': 0.0,
            'sad': 0.0,
            'anger': 0.0,
            'surprise': 0.0,
            'fear': 0.0,
            'neutral': 0.0,
            'disgust': 0.0
        }
        
        # Happy: High pitch, high spectral centroid, moderate energy, fast tempo
        if pitch > 180 and spectral_centroid > 2000 and rms_energy > 0.05 and tempo > 100:
            emotion_scores['happy'] += 3
        if mfcc_mean[1] > 0 and harmonics > 2:  # Positive MFCC coefficient 1 + harmonics
            emotion_scores['happy'] += 1
            
        # Sad: Low pitch, low spectral centroid, low energy, slow tempo
        if pitch < 130 and spectral_centroid < 1000 and rms_energy < 0.03 and tempo < 80:
            emotion_scores['sad'] += 3
        if mfcc_mean[0] < -5 and zero_crossing < 0.1:  # Negative MFCC + low zero crossing
            emotion_scores['sad'] += 1
            
        # Anger: High energy, high zero-crossing rate, moderate pitch, strong harmonics
        if rms_energy > 0.1 and zero_crossing > 0.15 and harmonics > 3:
            emotion_scores['anger'] += 3
        if mfcc_mean[2] > 2 and spectral_centroid > 2500:  # Specific MFCC pattern for anger
            emotion_scores['anger'] += 1
            
        # Surprise: High zero-crossing, moderate energy, high spectral variation
        if zero_crossing > 0.2 and rms_energy > 0.06 and features['spectral_centroid_std'] > 500:
            emotion_scores['surprise'] += 2
        if spectral_centroid > 2500 and tempo > 120:
            emotion_scores['surprise'] += 1
            
        # Fear: High pitch, low energy, high spectral centroid variation
        if pitch > 200 and rms_energy < 0.04 and features['spectral_centroid_std'] > 800:
            emotion_scores['fear'] += 2
        if np.std(features['mfcc_std']) > 2 and zero_crossing < 0.08:  # High MFCC variation + low ZCR
            emotion_scores['fear'] += 1
            
        # Disgust: Moderate energy, specific spectral characteristics
        if 0.04 < rms_energy < 0.08 and spectral_centroid < 1500 and zero_crossing > 0.12:
            emotion_scores['disgust'] += 2
        if mfcc_mean[3] < -1 and mfcc_mean[4] > 1:  # Specific MFCC pattern
            emotion_scores['disgust'] += 1
            
        # Neutral: Balanced features, moderate tempo
        if 0.02 < rms_energy < 0.08 and 120 < pitch < 180 and 80 < tempo < 120:
            emotion_scores['neutral'] += 2
        if np.std(mfcc_mean) < 1.5:  # Low MFCC variation
            emotion_scores['neutral'] += 1
            
        # Find dominant emotion
        max_score = max(emotion_scores.values())
        if max_score == 0:
            return "neutral", 0.5
            
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)
        confidence = min(0.95, max_score / 5.0)  # Normalize to 0-0.95 range
        
        return dominant_emotion, confidence
    
    def detect_emotion(self, audio_data):
        """Detect emotion using MFCC features"""
        features = self.extract_mfcc_features(audio_data)
        return self.analyze_emotion_from_features(features)
    
    def real_time_emotion_analysis(self, audio_chunk):
        """Real-time emotion analysis with temporal smoothing"""
        emotion, confidence = self.detect_emotion(audio_chunk)
        
        # Update emotion history for temporal smoothing
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        
        # Keep only recent history
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)
        
        # Apply temporal smoothing if we have enough history
        if len(self.emotion_history) >= 3:
            # Use weighted average based on confidence
            emotion_weights = {}
            for i, (emot, conf) in enumerate(zip(self.emotion_history, self.confidence_history)):
                weight = conf * (i + 1) / len(self.emotion_history)  # Recent frames have higher weight
                emotion_weights[emot] = emotion_weights.get(emot, 0) + weight
            
            # Find emotion with highest weighted score
            smoothed_emotion = max(emotion_weights, key=emotion_weights.get)
            smoothed_confidence = min(0.95, emotion_weights[smoothed_emotion] / sum(emotion_weights.values()))
            
            # Only update if confidence is better
            if smoothed_confidence > confidence * 0.8:
                emotion = smoothed_emotion
                confidence = smoothed_confidence
        
        return emotion, confidence
    
    def get_emotion_statistics(self):
        """Get statistics about recent emotion detections"""
        if not self.emotion_history:
            return {}
        
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        total_detections = len(self.emotion_history)
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        dominant_percentage = (emotion_counts[dominant_emotion] / total_detections) * 100
        
        return {
            'total_detections': total_detections,
            'dominant_emotion': dominant_emotion,
            'dominant_percentage': dominant_percentage,
            'emotion_distribution': emotion_counts,
            'average_confidence': np.mean(self.confidence_history) if self.confidence_history else 0
        }