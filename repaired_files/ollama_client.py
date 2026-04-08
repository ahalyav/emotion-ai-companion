"""
OllamaClient — with 120 s timeout, 3-attempt retry, stream=False,
and automatic Gemini fallback when all retries are exhausted.
"""

import requests
import json
import time
from config import Config


class OllamaClient:
    MAX_RETRIES    = 3
    RETRY_DELAY    = 2      # seconds between retries
    REQUEST_TIMEOUT = 120   # seconds (fixes Read timed out)

    def __init__(self):
        self.base_url      = Config.OLLAMA_BASE_URL
        self.model         = Config.OLLAMA_MODEL
        self.is_available  = self._check_availability()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def _check_availability(self) -> bool:
        """Quick ping to confirm Ollama is running."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                print("✅ Ollama is available")
                return True
            print("❌ Ollama not responding properly")
            return False
        except Exception as exc:
            print(f"❌ Ollama not available: {exc}")
            return False

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def _build_emotion_prompt(self, user_message: str, emotion_data: dict) -> str:
        """
        Build an autism-friendly emotion-aware prompt.
        Matches the format requested: "User emotion detected: X (confidence Y%)."
        """
        emotion    = emotion_data.get('current_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.0)

        context_map = {
            'happy':   "Respond in an uplifting, cheerful manner that matches their mood.",
            'sad':     "Respond with empathy, comfort, and support. Offer gentle encouragement.",
            'angry':   "Respond calmly and help de-escalate. Be understanding, not confrontational.",
            'surprise':"Respond with curiosity and engagement. Help them process the unexpected.",
            'fear':    "Respond with reassurance and safety. Help them feel more secure.",
            'disgust': "Respond with understanding and offer alternative perspectives.",
            'neutral': "Respond in a balanced, informative, and helpful manner.",
        }
        guidance = context_map.get(emotion, context_map['neutral'])
        pct = int(confidence * 100)

        return (
            f"User emotion detected: {emotion} (confidence {pct}%). "
            f"Give supportive feedback for an autistic user in simple language.\n\n"
            f"Context: {guidance}\n\n"
            f"User message: {user_message}\n\n"
            f"Response (keep it under 4 sentences, plain language):"
        )

    # ------------------------------------------------------------------
    # Core generation with retry loop
    # ------------------------------------------------------------------

    def _post_with_retry(self, prompt: str, max_tokens: int) -> str:
        """
        POST to Ollama with up to MAX_RETRIES attempts, 120 s timeout,
        and stream=False to prevent hanging connections.

        Returns the text response or raises RuntimeError after all retries.
        """
        payload = {
            "model":   self.model,
            "prompt":  prompt,
            "stream":  False,       # ← prevents streaming freeze
            "options": {
                "temperature":  0.7,
                "top_p":        0.9,
                "num_predict":  max_tokens,
            }
        }

        last_error = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT,  # 120 s
                    stream=False,                  # explicit, matches payload
                )
                if resp.status_code == 200:
                    return resp.json().get('response', '').strip()
                last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except requests.exceptions.Timeout as exc:
                last_error = f"Timeout on attempt {attempt}: {exc}"
                print(f"⚠️  Ollama {last_error}")
            except Exception as exc:
                last_error = f"Error on attempt {attempt}: {exc}"
                print(f"⚠️  Ollama {last_error}")

            if attempt < self.MAX_RETRIES:
                time.sleep(self.RETRY_DELAY)

        raise RuntimeError(f"Ollama failed after {self.MAX_RETRIES} attempts. Last error: {last_error}")

    # ------------------------------------------------------------------
    # Public API  (same signatures app.py uses)
    # ------------------------------------------------------------------

    def get_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Alias matching app.py calls."""
        return self.generate_response(prompt, {}, max_tokens)

    def generate_response(self, prompt: str,
                          emotion_data: dict,
                          max_tokens: int = 150) -> str:
        """
        Generate a response.  Falls back to Gemini automatically if Ollama
        is unavailable or all retries are exhausted.
        """
        if not self.is_available:
            return self._gemini_fallback(prompt, emotion_data)

        full_prompt = self._build_emotion_prompt(prompt, emotion_data)
        try:
            return self._post_with_retry(full_prompt, max_tokens)
        except RuntimeError as exc:
            print(f"❌ {exc} — switching to Gemini fallback")
            return self._gemini_fallback(prompt, emotion_data)

    # ------------------------------------------------------------------
    # Gemini / rule-based fallback
    # ------------------------------------------------------------------

    def _gemini_fallback(self, prompt: str, emotion_data: dict) -> str:
        """
        Try GeminiFallbackClient first; if that also fails, return a
        rule-based response so the UI always gets something useful.
        """
        try:
            from llm_companion.gemini_client import GeminiFallbackClient
            emotion    = emotion_data.get('current_emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.5)
            return GeminiFallbackClient().get_autism_feedback(emotion, confidence)
        except Exception:
            pass
        # Last-resort rule-based safeguard
        return _rule_based_response(
            emotion_data.get('current_emotion', 'neutral')
        )

    def create_emotion_aware_prompt(self, user_message: str,
                                    emotion_data: dict) -> str:
        """Legacy helper kept for backward compatibility."""
        return self._build_emotion_prompt(user_message, emotion_data)


# ---------------------------------------------------------------------------
# Shared rule-based safeguard (used by both OllamaClient and MockOllamaClient)
# ---------------------------------------------------------------------------

def _rule_based_response(emotion: str) -> str:
    rules = {
        'happy':   "You seem happy! That's great. 😊",
        'sad':     "It's okay to feel sad. Take a deep breath. You are not alone.",
        'angry':   "Try to relax and slow down. Take a few deep breaths.",
        'fear':    "You are safe right now. Breathe slowly — in for 4, out for 4.",
        'surprise':"Something unexpected happened. That's okay — take a moment.",
        'disgust': "It's okay to step away from what bothers you. You're in control.",
    }
    return rules.get(emotion, "Tell me how you're feeling. I'm here to listen.")


# ---------------------------------------------------------------------------
# Mock client for when Ollama is not running at all
# ---------------------------------------------------------------------------

class MockOllamaClient:
    """
    Rule-based fallback used when Ollama is unavailable.
    Extended with the autism-friendly safeguard responses.
    """

    def __init__(self):
        self.is_available = False

    def get_response(self, prompt: str) -> str:
        return self.generate_response(prompt, {})

    def generate_response(self, prompt: str,
                          emotion_data: dict,
                          max_tokens: int = 150) -> str:
        emotion    = emotion_data.get('current_emotion', 'neutral')
        responses = {
            'happy':   "I can tell you're feeling happy! That's wonderful. 😊 Keep spreading that positive energy!",
            'sad':     "It's okay to feel sad. I'm here for you. Take a slow breath — you are not alone. 💙",
            'angry':   "Try to relax and slow down. Take a few deep breaths — things will feel calmer soon.",
            'surprise':"Something unexpected happened! That's okay. Take a moment and look around slowly.",
            'fear':    "You are safe right now. Breathe in for 4 counts, hold, then out for 4. I'm here.",
            'disgust': "It's okay to step away from what bothers you. You're in control.",
            'neutral': "Tell me how you're feeling. I'm here to listen and support you.",
        }
        return responses.get(emotion, _rule_based_response(emotion))