import cv2
import numpy as np
import time
from config import Config


class SimpleFaceDetector:
    """
    Enhanced face-based emotion detector using multi-region feature analysis.
    Uses facial geometry regions + LBP texture + gradient energy scoring
    instead of raw brightness, yielding far more accurate emotion classification.
    """

    def __init__(self):
        self.labels = Config.EMOTION_LABELS['face']
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        # Optional eye cascade for face verification
        try:
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
        except Exception:
            self.eye_cascade = None

        self.last_analysis_time = 0
        self.analysis_interval = getattr(Config, 'FACE_ANALYSIS_INTERVAL', 0.5)
        self.emotion_history = []   # list of (emotion, confidence)
        self.max_history = 15       # longer window for better smoothing
        self.raw_scores_history = []  # for returning radar chart data

        # Per-emotion soft scores exposed to fusion engine / frontend
        self.last_scores = {e: 0.0 for e in self.labels}

    # ------------------------------------------------------------------
    # Low-level feature extractors
    # ------------------------------------------------------------------

    def _lbp_histogram(self, gray_region, radius=1, n_points=8, bins=16):
        """Compute a simplified LBP histogram for a grayscale region."""
        if gray_region is None or gray_region.size < 64:
            return np.zeros(bins)
        h, w = gray_region.shape
        lbp = np.zeros((h - 2 * radius, w - 2 * radius), dtype=np.uint8)
        center = gray_region[radius:-radius, radius:-radius].astype(np.float32)
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            dy = int(round(radius * np.sin(angle)))
            dx = int(round(radius * np.cos(angle)))
            yr = radius + dy
            xr = radius + dx
            # Clamp neighbour region
            yr_start = max(0, yr)
            yr_end = min(h, yr + h - 2 * radius)
            xr_start = max(0, xr)
            xr_end = min(w, xr + w - 2 * radius)
            neighbour_h = min(lbp.shape[0], yr_end - yr_start)
            neighbour_w = min(lbp.shape[1], xr_end - xr_start)
            neighbor = gray_region[yr_start:yr_start + neighbour_h,
                                   xr_start:xr_start + neighbour_w].astype(np.float32)
            lbp[:neighbour_h, :neighbour_w] |= (
                (neighbor >= center[:neighbour_h, :neighbour_w]).astype(np.uint8) << i
            )
        hist, _ = np.histogram(lbp.ravel(), bins=bins, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = hist.sum()
        if norm > 0:
            hist /= norm
        return hist

    def _gradient_energy(self, gray_region):
        """Compute mean gradient magnitude and its std — captures edge activity."""
        if gray_region is None or gray_region.size < 16:
            return 0.0, 0.0
        gx = cv2.Sobel(gray_region, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_region, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx ** 2 + gy ** 2)
        return float(np.mean(mag)), float(np.std(mag))

    def _region_stats(self, gray_region):
        """Return (mean, std, skew approx) of pixel intensities."""
        if gray_region is None or gray_region.size < 4:
            return 128.0, 0.0, 0.0
        flat = gray_region.astype(np.float32).ravel()
        mean = float(np.mean(flat))
        std = float(np.std(flat))
        # Approximate skewness via (mean - median)
        median = float(np.median(flat))
        skew = (mean - median) / (std + 1e-6)
        return mean, std, skew

    # ------------------------------------------------------------------
    # Multi-region facial feature extraction
    # ------------------------------------------------------------------

    def _extract_face_features(self, face_region_rgb):
        """
        Extract structured features from a detected face ROI.

        Regions:
          forehead  — top 25 %
          eyes      — 20-45 %
          nose      — 45-65 %
          mouth     — 65-85 %
          chin      — 85-100 %
        """
        h, w = face_region_rgb.shape[:2]
        gray = cv2.cvtColor(face_region_rgb, cv2.COLOR_RGB2GRAY)

        # Define slices
        slices = {
            'forehead': gray[0:int(h * 0.25), :],
            'eyes':     gray[int(h * 0.20):int(h * 0.45), :],
            'nose':     gray[int(h * 0.45):int(h * 0.65), :],
            'mouth':    gray[int(h * 0.65):int(h * 0.85), :],
            'chin':     gray[int(h * 0.85):, :],
        }

        feats = {}
        for name, region in slices.items():
            if region.size == 0:
                feats[name] = {
                    'mean': 128, 'std': 0, 'skew': 0,
                    'grad_mean': 0, 'grad_std': 0,
                    'lbp_uniformity': 0
                }
                continue
            mean, std, skew = self._region_stats(region)
            gmean, gstd = self._gradient_energy(region)
            lbp = self._lbp_histogram(region)
            # LBP uniformity: how concentrated the histogram is
            lbp_uniformity = float(np.max(lbp)) - float(np.min(lbp))
            feats[name] = {
                'mean': mean, 'std': std, 'skew': skew,
                'grad_mean': gmean, 'grad_std': gstd,
                'lbp_uniformity': lbp_uniformity
            }

        # Whole-face features
        feats['face_mean'], feats['face_std'], _ = self._region_stats(gray)
        feats['face_grad_mean'], feats['face_grad_std'] = self._gradient_energy(gray)

        # Asymmetry: compare left vs right halves of eyes/mouth bands
        eye_band = slices['eyes']
        left_eye = eye_band[:, :w // 2]
        right_eye = eye_band[:, w // 2:]
        feats['eye_asymmetry'] = abs(
            float(np.mean(left_eye)) - float(np.mean(right_eye))
        ) if left_eye.size and right_eye.size else 0.0

        mouth_band = slices['mouth']
        left_m = mouth_band[:, :w // 2]
        right_m = mouth_band[:, w // 2:]
        feats['mouth_asymmetry'] = abs(
            float(np.mean(left_m)) - float(np.mean(right_m))
        ) if left_m.size and right_m.size else 0.0

        # Mouth-to-forehead brightness ratio (smiling usually brightens lower face)
        feats['mouth_forehead_ratio'] = (
            feats['mouth']['mean'] / (feats['forehead']['mean'] + 1e-6)
        )

        return feats

    # ------------------------------------------------------------------
    # Emotion scoring based on features
    # ------------------------------------------------------------------

    def _score_emotions(self, feats):
        """
        Return a dict of {emotion: score} using rule-based multi-feature scoring.
        Scores are non-negative; higher = more likely.
        """
        scores = {e: 0.0 for e in self.labels}
        if not feats:
            scores['neutral'] = 1.0
            return scores

        mouth = feats['mouth']
        eyes = feats['eyes']
        forehead = feats['forehead']
        nose = feats['nose']
        face_grad = feats['face_grad_mean']
        face_std = feats['face_std']
        eye_asym = feats['eye_asymmetry']
        mouth_asym = feats['mouth_asymmetry']
        m2f = feats['mouth_forehead_ratio']

        # ---- HAPPY ----
        # Bright mouth region (cheeks rise + white teeth), high gradient in mouth,
        # relatively symmetric, moderate overall contrast
        if m2f > 1.05:
            scores['happy'] += 1.5
        if mouth['grad_mean'] > 25:
            scores['happy'] += 1.0
        if mouth['std'] > 35:
            scores['happy'] += 0.8
        if mouth_asym < 10 and eye_asym < 15:
            scores['happy'] += 0.5
        if face_std > 40 and face_grad > 18:
            scores['happy'] += 0.4

        # ---- SAD ----
        # Dark/low-contrast mouth, low gradient activity overall,
        # slightly asymmetric mouth (drooping corners)
        if m2f < 0.95:
            scores['sad'] += 1.5
        if face_grad < 12:
            scores['sad'] += 1.0
        if mouth['grad_mean'] < 15:
            scores['sad'] += 0.8
        if mouth['std'] < 20:
            scores['sad'] += 0.6
        if feats['face_mean'] < 90:
            scores['sad'] += 0.5

        # ---- ANGRY ----
        # High gradient in forehead (furrowed brow), high overall gradient,
        # high LBP uniformity in eye/forehead region (wrinkles/tension)
        if forehead['grad_mean'] > 28:
            scores['angry'] += 1.5
        if face_grad > 25:
            scores['angry'] += 1.0
        if forehead['lbp_uniformity'] > 0.15:
            scores['angry'] += 0.8
        if eyes['grad_mean'] > 22 and eyes['std'] > 30:
            scores['angry'] += 0.7
        if face_std > 50:
            scores['angry'] += 0.4

        # ---- SURPRISE ----
        # High eye-region gradient (wide open eyes = bright sclera),
        # high forehead activity (raised brows), wide overall std
        if eyes['grad_mean'] > 30:
            scores['surprise'] += 1.5
        if forehead['grad_mean'] > 22 and forehead['std'] > 35:
            scores['surprise'] += 1.2
        if face_std > 55:
            scores['surprise'] += 0.8
        if mouth['grad_mean'] > 28:   # open mouth
            scores['surprise'] += 0.5

        # ---- FEAR ----
        # Very high eye gradient (wide eyes), low mouth activity,
        # low overall brightness, high asymmetry
        if eyes['grad_mean'] > 28 and eyes['lbp_uniformity'] > 0.18:
            scores['fear'] += 1.5
        if feats['face_mean'] < 80:
            scores['fear'] += 0.8
        if mouth_asym > 18 or eye_asym > 20:
            scores['fear'] += 1.0
        if mouth['grad_mean'] < 18:
            scores['fear'] += 0.5

        # ---- DISGUST ----
        # Nose region high gradient (wrinkled nose), mouth asymmetry,
        # mid-range overall brightness
        if nose['grad_mean'] > 20:
            scores['disgust'] += 1.5
        if nose['lbp_uniformity'] > 0.14:
            scores['disgust'] += 0.8
        if mouth_asym > 15:
            scores['disgust'] += 0.8
        if 80 < feats['face_mean'] < 150 and face_std < 40:
            scores['disgust'] += 0.5

        # ---- NEUTRAL ----
        # Balanced, low-gradient face, low asymmetry, moderate brightness
        if face_grad < 18 and face_std < 38:
            scores['neutral'] += 1.5
        if eye_asym < 8 and mouth_asym < 8:
            scores['neutral'] += 1.0
        if 100 < feats['face_mean'] < 170:
            scores['neutral'] += 0.5
        if mouth['grad_mean'] < 20 and forehead['grad_mean'] < 20:
            scores['neutral'] += 0.5

        return scores

    # ------------------------------------------------------------------
    # Face detection & emotion pipeline
    # ------------------------------------------------------------------

    def detect_faces(self, frame):
        """Detect faces using Haar cascade with improved parameters."""
        if frame is None:
            return []
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)   # improve contrast for detection
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,             # lowered for better recall
                minSize=(40, 40),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            return faces if len(faces) else []
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def smooth_emotion(self, new_emotion, new_confidence):
        """Confidence-weighted temporal smoothing over recent history."""
        self.emotion_history.append((new_emotion, new_confidence))
        if len(self.emotion_history) > self.max_history:
            self.emotion_history.pop(0)
        if len(self.emotion_history) < 3:
            return new_emotion, new_confidence

        # Weighted vote: recent frames + higher-confidence frames win
        emotion_scores = {}
        n = len(self.emotion_history)
        for i, (emot, conf) in enumerate(self.emotion_history):
            # Recency weight: linearly increasing
            recency = (i + 1) / n
            weight = conf * recency
            emotion_scores[emot] = emotion_scores.get(emot, 0.0) + weight

        best_emotion = max(emotion_scores, key=emotion_scores.get)
        total = sum(emotion_scores.values())
        best_conf = emotion_scores[best_emotion] / total if total > 0 else new_confidence
        return best_emotion, min(0.95, best_conf)

    def detect_emotion(self, frame):
        """
        Full pipeline: detect face → extract features → score emotions →
        smooth over time. Returns (emotion_str, confidence, faces_list).
        """
        current_time = time.time()

        # Throttle heavy analysis
        if current_time - self.last_analysis_time < self.analysis_interval:
            if self.emotion_history:
                last_emot, last_conf = self.emotion_history[-1]
                return last_emot, last_conf, []
            return 'neutral', 0.3, []

        self.last_analysis_time = current_time

        if frame is None:
            return 'neutral', 0.3, []

        try:
            faces = self.detect_faces(frame)

            if len(faces) == 0:
                # No face — decay towards neutral with low confidence
                self.emotion_history.append(('neutral', 0.25))
                if len(self.emotion_history) > self.max_history:
                    self.emotion_history.pop(0)
                self.last_scores = {e: 0.0 for e in self.labels}
                self.last_scores['neutral'] = 1.0
                return 'neutral', 0.25, []

            # Use the largest face
            face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = face
            face_region = frame[y:y + h, x:x + w]

            feats = self._extract_face_features(face_region)
            raw_scores = self._score_emotions(feats)

            # Store normalised scores for radar chart
            total = sum(raw_scores.values()) or 1.0
            self.last_scores = {e: raw_scores[e] / total for e in self.labels}

            dominant = max(raw_scores, key=raw_scores.get)
            max_score = raw_scores[dominant]
            # Normalise confidence to [0.3, 0.92]
            confidence = min(0.92, 0.3 + (max_score / (max_score + 3.0)) * 0.62)

            smoothed_emotion, smoothed_confidence = self.smooth_emotion(
                dominant, confidence
            )
            return smoothed_emotion, smoothed_confidence, [face]

        except Exception as e:
            print(f"Emotion detection error: {e}")
            return 'neutral', 0.3, []

    def get_raw_scores(self):
        """Return dict of normalised 7-class probabilities (for radar chart)."""
        return dict(self.last_scores)