import numpy as np
import librosa
from config import Config


class MFCCEmotionDetector:
    """
    Enhanced audio emotion detector using:
    - MFCC (13 coefficients) + delta + delta-delta
    - Mel spectrogram band energies
    - Spectral flux (frame-to-frame change)
    - Voiced/unvoiced ratio
    - Pitch, harmonics, tempo (improved)
    - All features normalised to [0,1] before scoring
    - Exponential moving average (EMA) for temporal smoothing
    """

    EMOTION_LABELS = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    def __init__(self):
        self.labels = Config.EMOTION_LABELS.get('audio', self.EMOTION_LABELS)
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.mfcc_features = 13
        self.max_history = getattr(Config, 'EMOTION_HISTORY_DEPTH', 20)
        self.ema_alpha = getattr(Config, 'AUDIO_SMOOTHING_ALPHA', 0.3)

        self.emotion_history = []
        self.confidence_history = []
        # EMA state: dict {emotion: ema_score}
        self._ema_scores = {e: 0.0 for e in self.EMOTION_LABELS}

        # Raw 7-class soft scores exposed for radar chart
        self.last_scores = {e: 0.0 for e in self.EMOTION_LABELS}

    # ------------------------------------------------------------------
    # Feature extraction helpers
    # ------------------------------------------------------------------

    def _safe_normalise(self, value, lo, hi):
        """Linearly map value from [lo, hi] → [0, 1], clipped."""
        return float(np.clip((value - lo) / (hi - lo + 1e-9), 0.0, 1.0))

    def _extract_mel_band_energy(self, audio_data, n_mels=64):
        """Return mean energy in 4 sub-bands of a Mel spectrogram."""
        try:
            mel = librosa.feature.melspectrogram(
                y=audio_data, sr=self.sample_rate,
                n_mels=n_mels, hop_length=512
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            band_size = n_mels // 4
            bands = []
            for i in range(4):
                bands.append(float(np.mean(mel_db[i * band_size:(i + 1) * band_size])))
            return bands   # [low, mid-low, mid-high, high]
        except Exception:
            return [0.0, 0.0, 0.0, 0.0]

    def _spectral_flux(self, audio_data, hop_length=512):
        """Mean spectral flux — measures how fast the spectrum changes."""
        try:
            stft = np.abs(librosa.stft(audio_data, hop_length=hop_length))
            diff = np.diff(stft, axis=1)
            flux = np.mean(np.sum(diff ** 2, axis=0))
            return float(flux)
        except Exception:
            return 0.0

    def _voiced_ratio(self, audio_data, hop_length=512, energy_threshold=0.001):
        """Fraction of frames above energy threshold (voiced frames)."""
        try:
            rms = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
            threshold = np.max(rms) * energy_threshold
            return float(np.mean(rms > threshold))
        except Exception:
            return 0.5

    def estimate_pitch(self, audio_data):
        try:
            corr = np.correlate(audio_data, audio_data, mode='full')
            corr = corr[len(corr) // 2:]
            min_lag = int(self.sample_rate / 500)
            max_lag = int(self.sample_rate / 60)
            if len(corr) > max_lag and max_lag > min_lag:
                seg = corr[min_lag:max_lag]
                if len(seg) > 0:
                    idx = np.argmax(seg) + min_lag
                    return self.sample_rate / idx if idx > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0

    def detect_harmonics(self, audio_data):
        try:
            fft = np.abs(np.fft.rfft(audio_data))
            freqs = np.fft.rfftfreq(len(audio_data), 1 / self.sample_rate)
            fund_idx = np.argmax(fft[:len(fft) // 2]) if len(fft) > 2 else 0
            fund_freq = freqs[fund_idx]
            count = 0
            for i in range(2, 6):
                hf = fund_freq * i
                if hf >= freqs[-1]:
                    break
                region = (freqs > hf * 0.9) & (freqs < hf * 1.1)
                if np.any(region) and np.max(fft[region]) > np.max(fft) * 0.1:
                    count += 1
            return count
        except Exception:
            return 0

    def estimate_tempo(self, audio_data):
        try:
            if len(audio_data) < 4096:
                return 0.0
            onset_frames = librosa.onset.onset_detect(
                y=audio_data, sr=self.sample_rate, hop_length=512
            )
            if len(onset_frames) > 1:
                times = librosa.frames_to_time(onset_frames, sr=self.sample_rate, hop_length=512)
                intervals = np.diff(times)
                avg = np.mean(intervals)
                return 60.0 / avg if avg > 0 else 0.0
            return 0.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Main feature extraction
    # ------------------------------------------------------------------

    def extract_mfcc_features(self, audio_data):
        if audio_data is None or len(audio_data) < 512:
            return None
        try:
            # Ensure reasonable length
            max_samples = 3 * self.sample_rate
            if len(audio_data) < 2048:
                audio_data = np.pad(audio_data, (0, 2048 - len(audio_data)))
            else:
                audio_data = audio_data[:max_samples]

            # MFCC
            mfccs = librosa.feature.mfcc(
                y=audio_data, sr=self.sample_rate,
                n_mfcc=self.mfcc_features, n_fft=1024, hop_length=512
            )
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio_data, sr=self.sample_rate)
            zcr = librosa.feature.zero_crossing_rate(audio_data)
            rms = librosa.feature.rms(y=audio_data)
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio_data, sr=self.sample_rate)
            chroma = librosa.feature.chroma_stft(
                y=audio_data, sr=self.sample_rate)

            feats = {
                'mfcc_mean': np.mean(mfccs, axis=1),
                'mfcc_std': np.std(mfccs, axis=1),
                'mfcc_delta': np.mean(librosa.feature.delta(mfccs), axis=1),
                'mfcc_delta2': np.mean(librosa.feature.delta(mfccs, order=2), axis=1),
                'spectral_centroid_mean': float(np.mean(spectral_centroid)),
                'spectral_centroid_std': float(np.std(spectral_centroid)),
                'zero_crossing_mean': float(np.mean(zcr)),
                'rms_mean': float(np.mean(rms)),
                'rms_std': float(np.std(rms)),
                'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
                'chroma_mean': np.mean(chroma, axis=1),
                'pitch': self.estimate_pitch(audio_data),
                'harmonics': self.detect_harmonics(audio_data),
                'tempo': self.estimate_tempo(audio_data),
                # New features
                'mel_bands': self._extract_mel_band_energy(audio_data),
                'spectral_flux': self._spectral_flux(audio_data),
                'voiced_ratio': self._voiced_ratio(audio_data),
            }
            return feats
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None

    # ------------------------------------------------------------------
    # Emotion scoring (normalised features)
    # ------------------------------------------------------------------

    def _build_soft_scores(self, feats):
        """
        Return a {emotion: score} dict using normalised features.
        All features are mapped to [0,1] so thresholds are scale-independent.
        """
        scores = {e: 0.0 for e in self.EMOTION_LABELS}
        if feats is None:
            scores['neutral'] = 1.0
            return scores

        sn = self._safe_normalise

        pitch = feats['pitch']
        rms = feats['rms_mean']
        zcr = feats['zero_crossing_mean']
        centroid = feats['spectral_centroid_mean']
        harmonics = feats['harmonics']
        tempo = feats['tempo']
        flux = feats['spectral_flux']
        voiced = feats['voiced_ratio']
        mfcc = feats['mfcc_mean']
        rms_std = feats['rms_std']
        bands = feats['mel_bands']          # [low, mid-low, mid-high, high]

        # Normalised scalars
        n_pitch = sn(pitch, 50, 400)        # 50-400 Hz
        n_rms = sn(rms, 0, 0.3)
        n_zcr = sn(zcr, 0, 0.5)
        n_centroid = sn(centroid, 500, 6000)
        n_tempo = sn(tempo, 40, 200)
        n_flux = sn(flux, 0, 5000)
        n_rms_std = sn(rms_std, 0, 0.1)
        n_voiced = voiced                   # already [0,1]
        n_high_band = sn(bands[3], -80, 0)  # high freq energy (dB)
        n_low_band = sn(bands[0], -80, 0)

        # ---- HAPPY ----
        scores['happy'] += n_pitch * 1.5       # high pitch
        scores['happy'] += n_rms * 1.2         # energetic
        scores['happy'] += n_centroid * 1.0    # bright timbre
        scores['happy'] += n_tempo * 0.8       # fast tempo
        scores['happy'] += n_voiced * 0.5      # well-voiced
        if mfcc[1] > 0:
            scores['happy'] += 0.4             # positive 1st delta MFCC

        # ---- SAD ----
        scores['sad'] += (1 - n_pitch) * 1.5   # low pitch
        scores['sad'] += (1 - n_rms) * 1.2     # soft/quiet
        scores['sad'] += (1 - n_centroid) * 0.8
        scores['sad'] += (1 - n_tempo) * 0.8
        scores['sad'] += n_low_band * 0.5      # low-frequency dominant
        if mfcc[0] < -3:
            scores['sad'] += 0.5

        # ---- ANGER ----
        scores['angry'] += n_rms * 1.8          # loud
        scores['angry'] += n_zcr * 1.2          # noisy, harsh
        scores['angry'] += n_flux * 1.0         # fast changing spectrum
        scores['angry'] += n_rms_std * 0.8      # high energy variance
        if harmonics >= 3:
            scores['angry'] += 0.6
        if mfcc[2] > 1.5:
            scores['angry'] += 0.4

        # ---- SURPRISE ----
        scores['surprise'] += n_flux * 1.5      # sudden spectral change
        scores['surprise'] += n_zcr * 0.9
        scores['surprise'] += n_high_band * 0.8
        scores['surprise'] += n_voiced * 0.4
        scores['surprise'] += n_tempo * 0.5

        # ---- FEAR ----
        scores['fear'] += n_pitch * 1.2         # high pitch (trembling)
        scores['fear'] += (1 - n_rms) * 1.0    # soft/breathless
        scores['fear'] += n_rms_std * 1.0       # unsteady energy
        scores['fear'] += (1 - n_voiced) * 0.8  # more unvoiced (breaths)
        if feats['spectral_centroid_std'] > 600:
            scores['fear'] += 0.6

        # ---- DISGUST ----
        scores['disgust'] += (1 - n_centroid) * 0.8
        scores['disgust'] += n_low_band * 0.8
        scores['disgust'] += (1 - n_tempo) * 0.5
        if mfcc[4] > 1.0:
            scores['disgust'] += 0.5
        if 0.3 < n_rms < 0.6:
            scores['disgust'] += 0.4

        # ---- NEUTRAL ----
        # Neutral = moderate across most dimensions
        neutrality = 1.0 - max(
            abs(n_pitch - 0.35),
            abs(n_rms - 0.25),
            abs(n_centroid - 0.35),
            abs(n_tempo - 0.35),
        )
        scores['neutral'] += max(0.0, neutrality) * 2.0
        if n_zcr < 0.3 and n_rms_std < 0.3:
            scores['neutral'] += 0.5

        return scores

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_emotion(self, audio_data):
        feats = self.extract_mfcc_features(audio_data)
        raw_scores = self._build_soft_scores(feats)

        # Normalise scores → probabilities
        total = sum(raw_scores.values()) or 1.0
        norm_scores = {e: raw_scores[e] / total for e in self.EMOTION_LABELS}
        self.last_scores = norm_scores

        dominant = max(norm_scores, key=norm_scores.get)
        confidence = min(0.93, norm_scores[dominant] * 2.0)  # scale for display
        return dominant, confidence

    def real_time_emotion_analysis(self, audio_chunk):
        """Real-time analysis with EMA smoothing across history."""
        emotion, confidence = self.detect_emotion(audio_chunk)

        # Update history
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
            self.confidence_history.pop(0)

        # EMA update on raw scores
        for e in self.EMOTION_LABELS:
            self._ema_scores[e] = (
                self.ema_alpha * self.last_scores.get(e, 0.0)
                + (1 - self.ema_alpha) * self._ema_scores.get(e, 0.0)
            )

        # Use EMA scores for smoothed prediction
        smoothed_emotion = max(self._ema_scores, key=self._ema_scores.get)
        ema_total = sum(self._ema_scores.values()) or 1.0
        smoothed_conf = min(0.93, self._ema_scores[smoothed_emotion] / ema_total * 2.0)

        return smoothed_emotion, smoothed_conf

    def get_raw_scores(self):
        """Return EMA-smoothed 7-class probabilities for radar chart."""
        total = sum(self._ema_scores.values()) or 1.0
        return {e: self._ema_scores[e] / total for e in self.EMOTION_LABELS}

    def get_emotion_statistics(self):
        if not self.emotion_history:
            return {}
        counts = {}
        for e in self.emotion_history:
            counts[e] = counts.get(e, 0) + 1
        total = len(self.emotion_history)
        dominant = max(counts, key=counts.get)
        return {
            'total_detections': total,
            'dominant_emotion': dominant,
            'dominant_percentage': counts[dominant] / total * 100,
            'emotion_distribution': counts,
            'average_confidence': float(np.mean(self.confidence_history))
            if self.confidence_history else 0.0,
        }