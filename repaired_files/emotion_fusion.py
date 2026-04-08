import numpy as np
from collections import deque
from config import Config


class EmotionFusion:
    """
    Enhanced multi-modal emotion fusion using Bayesian soft-score merging.

    Weights: audio=0.6, video=0.4 by default.
    Dynamic rebalancing: if audio_confidence < 0.4, swap to video=0.7 / audio=0.3.
    Output smoothed over the last 10 fused predictions to prevent flickering.
    """

    EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

    # Audio labels → canonical labels
    AUDIO_MAP = {
        'anger': 'angry', 'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fear',
        'happy': 'happy',
        'sad': 'sad',
        'surprise': 'surprise',
        'neutral': 'neutral',
    }

    # Low-audio threshold: if audio confidence falls below this value,
    # video is treated as the primary signal.
    LOW_AUDIO_THRESHOLD = 0.4

    # Smoothing buffer depth
    SMOOTH_WINDOW = 10

    def __init__(self):
        # Use fixed weights (audio=0.6, video=0.4) as specified.
        # Config values are used as a fallback only.
        self.audio_weight = 0.6
        self.face_weight  = 0.4

        self.emotion_labels = Config.EMOTION_LABELS.get('combined', self.EMOTION_LABELS)
        self._last_fused_scores = {e: 0.0 for e in self.EMOTION_LABELS}

        # Rolling smoothing buffer: stores (emotion, confidence) tuples
        self._smooth_buffer: deque = deque(maxlen=self.SMOOTH_WINDOW)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _uniform_vector(self):
        n = len(self.EMOTION_LABELS)
        return {e: 1.0 / n for e in self.EMOTION_LABELS}

    def _one_hot(self, emotion, confidence, is_audio=False):
        """
        Build a soft probability vector from a single (emotion, confidence) pair.
        The detected emotion gets `confidence` probability; the remainder is
        spread uniformly across the other classes.
        """
        canonical = self.AUDIO_MAP.get(emotion, emotion) if is_audio else emotion
        if canonical not in self.EMOTION_LABELS:
            canonical = 'neutral'
        n = len(self.EMOTION_LABELS)
        residual = (1.0 - confidence) / (n - 1)
        vec = {e: residual for e in self.EMOTION_LABELS}
        vec[canonical] = confidence
        return vec

    def _weighted_sum(self, audio_vec, face_vec, audio_w, face_w):
        """Weighted sum of two probability vectors, then normalise to sum=1."""
        merged = {}
        for e in self.EMOTION_LABELS:
            merged[e] = audio_w * audio_vec.get(e, 0.0) + face_w * face_vec.get(e, 0.0)
        total = sum(merged.values()) or 1.0
        return {e: merged[e] / total for e in self.EMOTION_LABELS}

    def _compute_weights(self, audio_confidence):
        """
        Return (audio_w, face_w) based on audio confidence.

        If audio_confidence < LOW_AUDIO_THRESHOLD, video dominates (0.7/0.3).
        Otherwise use the default spec weights (0.6/0.4).
        """
        if audio_confidence < self.LOW_AUDIO_THRESHOLD:
            return 0.3, 0.7   # video dominates
        return self.audio_weight, self.face_weight   # default 0.6 / 0.4

    def _smooth_output(self, emotion: str, confidence: float):
        """
        Confidence-weighted majority vote over the last SMOOTH_WINDOW predictions.
        Prevents the final fused output from flickering between emotions.
        """
        self._smooth_buffer.append((emotion, confidence))

        tally: dict = {}
        total_weight = 0.0
        for e, c in self._smooth_buffer:
            tally[e] = tally.get(e, 0.0) + c
            total_weight += c

        best = max(tally, key=tally.get)
        best_conf = tally[best] / (total_weight or 1.0)
        return best, round(min(0.97, best_conf), 3)

    # ------------------------------------------------------------------
    # Public API (same signatures as original — no app.py changes needed)
    # ------------------------------------------------------------------

    def fuse_emotions(self, audio_emotion, audio_confidence,
                      face_emotion, face_confidence,
                      audio_scores=None, face_scores=None):
        """
        Fuse two modalities into a single (emotion, confidence) pair.

        - Uses raw score dicts when available (more accurate).
        - Falls back to soft one-hot vectors from scalar (emotion, confidence).
        - Applies dynamic weight rebalancing based on audio_confidence.
        - Smoothes output over last 10 fused predictions.
        """
        audio_unavail = audio_emotion == 'unavailable'
        face_unavail  = face_emotion  == 'unavailable'

        if audio_unavail and face_unavail:
            return 'neutral', 0.0

        if audio_unavail:
            return face_emotion, face_confidence

        if face_unavail:
            canonical = self.AUDIO_MAP.get(audio_emotion, audio_emotion)
            return canonical, audio_confidence

        # ── Dynamic weight selection ──────────────────────────────────
        audio_w, face_w = self._compute_weights(audio_confidence)

        # ── Build probability vectors ─────────────────────────────────
        if audio_scores and len(audio_scores) >= 7:
            a_vec = {}
            for k, v in audio_scores.items():
                canonical_k = self.AUDIO_MAP.get(k, k)
                if canonical_k in self.EMOTION_LABELS:
                    a_vec[canonical_k] = a_vec.get(canonical_k, 0.0) + v
            for e in self.EMOTION_LABELS:
                a_vec.setdefault(e, 0.0)
        else:
            a_vec = self._one_hot(audio_emotion, audio_confidence, is_audio=True)

        if face_scores and len(face_scores) >= 7:
            f_vec = {e: face_scores.get(e, 0.0) for e in self.EMOTION_LABELS}
        else:
            f_vec = self._one_hot(face_emotion, face_confidence, is_audio=False)

        # ── Weighted merge ────────────────────────────────────────────
        merged = self._weighted_sum(a_vec, f_vec, audio_w, face_w)
        self._last_fused_scores = merged

        dominant = max(merged, key=merged.get)
        dom_prob = merged[dominant]

        # Agreement bonus
        a_dominant = self.AUDIO_MAP.get(audio_emotion, audio_emotion)
        agree = (a_dominant == face_emotion)
        if agree:
            raw_confidence = min(0.97, dom_prob * 1.25)
        else:
            gap = abs(audio_confidence - face_confidence)
            penalty = 0.85 if gap < 0.15 else 1.0
            raw_confidence = min(0.94, dom_prob * 1.1 * penalty)

        # ── Temporal smoothing over last 10 predictions ───────────────
        smoothed_emotion, smoothed_confidence = self._smooth_output(
            dominant, raw_confidence
        )
        return smoothed_emotion, smoothed_confidence

    def get_fusion_analysis(self, audio_data, face_data,
                            audio_scores=None, face_scores=None):
        """Comprehensive fusion analysis dict (same signature as original)."""
        audio_emotion    = audio_data.get('emotion', 'neutral')
        audio_confidence = audio_data.get('confidence', 0.0)
        face_emotion     = face_data.get('emotion', 'neutral')
        face_confidence  = face_data.get('confidence', 0.0)

        fused_emotion, fused_confidence = self.fuse_emotions(
            audio_emotion, audio_confidence,
            face_emotion, face_confidence,
            audio_scores=audio_scores,
            face_scores=face_scores,
        )

        audio_canonical = self.AUDIO_MAP.get(audio_emotion, audio_emotion)
        return {
            'fused_emotion':      fused_emotion,
            'fused_confidence':   fused_confidence,
            'audio_emotion':      audio_canonical,
            'audio_confidence':   audio_confidence,
            'face_emotion':       face_emotion,
            'face_confidence':    face_confidence,
            'modality_agreement': audio_canonical == face_emotion,
            'fusion_method':      'bayesian_soft_merge_v2',
            'fused_scores':       self._last_fused_scores,
        }

    def get_raw_scores(self):
        """Return the last fused 7-class probability vector for radar chart."""
        return dict(self._last_fused_scores)

    def get_session_summary(self, session_data):
        if not session_data:
            return {}
        emotions = session_data.get('emotions', [])
        if not emotions:
            return {}
        counts = {}
        for snap in emotions:
            e = snap.get('emotion', 'neutral')
            counts[e] = counts.get(e, 0) + 1
        dominant = max(counts, key=counts.get)
        total = len(emotions)
        return {
            'total_frames':        total,
            'dominant_emotion':    dominant,
            'dominant_percentage': counts[dominant] / total * 100,
            'emotion_distribution': counts,
            'session_duration':    session_data.get('duration', 0),
        }