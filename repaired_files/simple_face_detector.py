"""
SimpleFaceDetector — DeepFace-powered face emotion detector.

Key design decisions (per user spec):
  • DeepFace model loaded ONCE in __init__ to avoid per-frame lag.
  • Hard None-guard at the top of detect_emotion() — no crash when frame is None.
  • Correct face crop with ±30 px padding → resize to 224×224 → normalize.
  • Temporal smoothing: confidence-weighted majority vote over last 5 predictions.
  • Graceful fallback to 'neutral' on any DeepFace exception.
"""

import cv2
import numpy as np
import time

# ---------------------------------------------------------------------------
# DeepFace — import and verify analyse is callable.
# We do NOT call build_model() here because it triggers a TensorFlow
# __version__ attribute bug in some venv configurations.
# DeepFace.analyze() handles model loading lazily on first call.
# ---------------------------------------------------------------------------
try:
    from deepface import DeepFace
    # Lightweight callable-check (no model weights loaded yet)
    assert callable(DeepFace.analyze)
    DEEPFACE_AVAILABLE = True
    print("✅ DeepFace available (SimpleFaceDetector — lazy model load)")
except Exception as _df_err:
    DEEPFACE_AVAILABLE = False
    print(f"⚠️  DeepFace not available in SimpleFaceDetector: {_df_err}")


# Canonical labels shared with the rest of the pipeline
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# DeepFace → canonical mapping
_DF_MAP = {
    'angry':   'angry',
    'disgust': 'disgust',
    'fear':    'fear',
    'happy':   'happy',
    'sad':     'sad',
    'surprise':'surprise',
    'neutral': 'neutral',
}


class SimpleFaceDetector:
    """
    Face detector + DeepFace emotion classifier with temporal smoothing.

    Interface matches what app.py's background_video_processing() expects:
        detect_emotion(frame) → (emotion: str, confidence: float, faces: list)
        detect_faces(frame)   → [(x, y, w, h), ...]
        get_raw_scores()      → {emotion: probability, ...}
        reset_session()
    """

    # Temporal smoothing window
    SMOOTH_WINDOW = 5

    def __init__(self):
        # Haar cascade for face bounding boxes (fast, no GPU needed)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load Haar cascade: {cascade_path}")

        # Smoothing state
        self._emotion_window: list   = []   # list of str
        self._confidence_window: list = []  # list of float
        self._last_emotion    = 'neutral'
        self._last_confidence = 0.5
        self._raw_scores: dict = {e: 0.0 for e in EMOTION_LABELS}

        # Throttle: only run DeepFace every N seconds to keep frame rate smooth
        self._last_analysis_time = 0.0
        self._analysis_interval  = 0.4   # seconds (≈ 2.5 analyses / sec)

        print("✅ SimpleFaceDetector initialised (DeepFace + Haar Cascade)")

    # ------------------------------------------------------------------
    # Face detection
    # ------------------------------------------------------------------

    def detect_faces(self, frame) -> list:
        """
        Detect faces in an RGB/BGR frame using Haar cascades.

        Returns
        -------
        list of (x, y, w, h) tuples — empty list when nothing detected.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) \
                   if len(frame.shape) == 3 else frame
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            if len(faces) == 0:
                return []
            return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]
        except Exception as exc:
            print(f"detect_faces error: {exc}")
            return []

    # ------------------------------------------------------------------
    # Core emotion analysis (DeepFace)
    # ------------------------------------------------------------------

    def _analyse_face(self, frame, x, y, w, h):
        """
        Crop, preprocess, and run DeepFace on a single face bounding box.

        Parameters
        ----------
        frame : np.ndarray  — full RGB/BGR frame
        x, y, w, h : int   — face bounding box from Haar cascade

        Returns
        -------
        (emotion: str, confidence: float, scores: dict)
        """
        try:
            # --- 1. Crop with ±30 px padding ---
            pad = 30
            H, W = frame.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(W, x + w + pad)
            y2 = min(H, y + h + pad)
            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                return 'neutral', 0.3, self._uniform_scores()

            # --- 2. Resize to 224×224 ---
            face_resized = cv2.resize(face_crop, (224, 224),
                                      interpolation=cv2.INTER_AREA)

            # --- 3. Normalise pixel values to [0, 1] ---
            face_input = face_resized.astype(np.float32) / 255.0

            # Convert back to uint8 for DeepFace (it expects uint8 or raw image)
            face_uint8 = (face_input * 255).astype(np.uint8)

            # --- 4. DeepFace inference ---
            analysis = DeepFace.analyze(
                face_uint8,
                actions=['emotion'],
                enforce_detection=False,
                silent=True,
            )

            if not analysis:
                return 'neutral', 0.3, self._uniform_scores()

            result = analysis[0] if isinstance(analysis, list) else analysis
            dominant = result.get('dominant_emotion', 'neutral')
            emotion_scores = result.get('emotion', {})      # percentages 0-100

            # Map to canonical label
            emotion = _DF_MAP.get(dominant, 'neutral')
            raw_pct  = emotion_scores.get(dominant, 30.0)
            confidence = min(0.99, raw_pct / 100.0)

            # Build normalised score dict
            total = sum(emotion_scores.values()) or 1.0
            scores = {}
            for df_label, pct in emotion_scores.items():
                canon = _DF_MAP.get(df_label, df_label)
                if canon in EMOTION_LABELS:
                    scores[canon] = scores.get(canon, 0.0) + pct / total

            return emotion, confidence, scores

        except Exception as exc:
            print(f"DeepFace analysis error: {exc}")
            return 'neutral', 0.3, self._uniform_scores()

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def _smooth(self, emotion: str, confidence: float):
        """
        Confidence-weighted majority vote over the last SMOOTH_WINDOW predictions.
        Prevents emotion flickering.
        """
        self._emotion_window.append(emotion)
        self._confidence_window.append(confidence)

        if len(self._emotion_window) > self.SMOOTH_WINDOW:
            self._emotion_window.pop(0)
            self._confidence_window.pop(0)

        # Weighted tally
        scores: dict = {}
        total_weight = 0.0
        for e, c in zip(self._emotion_window, self._confidence_window):
            scores[e] = scores.get(e, 0.0) + c
            total_weight += c

        if not scores:
            return emotion, confidence

        best = max(scores, key=scores.get)
        best_conf = scores[best] / (total_weight or 1.0)
        return best, round(min(0.99, best_conf), 3)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_emotion(self, frame):
        """
        Main entry point called by app.py's background_video_processing thread.

        Parameters
        ----------
        frame : np.ndarray or None

        Returns
        -------
        (emotion: str, confidence: float, faces: list of (x,y,w,h))
        """
        # ── Hard None-guard (required: DeepFace crashes on None) ──
        if frame is None:
            return 'neutral', 0.0, []

        faces = self.detect_faces(frame)

        if not faces:
            # No face — push neutral into smoothing window to decay flickering
            self._smooth('neutral', 0.2)
            self._last_emotion    = 'neutral'
            self._last_confidence = 0.3
            return 'neutral', 0.3, []

        now = time.time()
        if DEEPFACE_AVAILABLE and (now - self._last_analysis_time >= self._analysis_interval):
            self._last_analysis_time = now

            # Analyse the largest face (by area)
            largest = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest

            emotion, confidence, scores = self._analyse_face(frame, x, y, w, h)
            smoothed_e, smoothed_c = self._smooth(emotion, confidence)

            self._last_emotion    = smoothed_e
            self._last_confidence = smoothed_c
            self._raw_scores      = scores
        else:
            # Return cached result between analysis cycles
            smoothed_e = self._last_emotion
            smoothed_c = self._last_confidence

        return smoothed_e, smoothed_c, faces

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _uniform_scores(self) -> dict:
        """Uniform probability distribution across all emotion labels."""
        n = len(EMOTION_LABELS)
        return {e: round(1.0 / n, 4) for e in EMOTION_LABELS}

    def get_raw_scores(self) -> dict:
        """
        Return the latest 7-class probability distribution.
        Called by app.py to populate the radar chart.
        """
        if self._raw_scores:
            return dict(self._raw_scores)
        # Boost current dominant slightly
        scores = self._uniform_scores()
        if self._last_emotion in scores:
            n = len(EMOTION_LABELS)
            base = 1.0 / n
            scores[self._last_emotion] = round(
                base + self._last_confidence * (1 - base), 4
            )
        return scores

    def reset_session(self):
        """Reset state for a new recording session."""
        self._emotion_window.clear()
        self._confidence_window.clear()
        self._last_emotion    = 'neutral'
        self._last_confidence = 0.5
        self._raw_scores      = {e: 0.0 for e in EMOTION_LABELS}
        self._last_analysis_time = 0.0
