"""
Gemini LLM Client — Autism-Friendly Emotional Companion
Uses Google Gemini API (gemini-1.5-flash) for empathetic, simple,
autism-friendly responses.

Falls back to GeminiFallbackClient when no API key is configured.
"""

import os
import re

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    print("⚠️  google-generativeai not found — installing automatically...")
    try:
        import subprocess, sys
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "google-generativeai"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        import google.generativeai as genai
        GENAI_AVAILABLE = True
        print("✅ google-generativeai installed successfully")
    except Exception as _install_err:
        print(f"❌ Could not install google-generativeai: {_install_err}")
        GENAI_AVAILABLE = False

try:
    from config import Config
    _API_KEY = Config.GEMINI_API_KEY
    _MODEL   = Config.GEMINI_MODEL
except Exception:
    _API_KEY = os.getenv('GEMINI_API_KEY', '')
    _MODEL   = 'gemini-1.5-flash'


# ---------------------------------------------------------------------------
# Autism-friendly system instruction (injected into every Gemini request)
# ---------------------------------------------------------------------------
AUTISM_SYSTEM_PROMPT = """You are an autism-friendly emotional support assistant.

Guidelines:
- Use very simple language (grade 4–6 reading level)
- Keep responses SHORT — maximum 4 sentences
- Avoid overwhelming text, metaphors, or idioms
- Be gentle, calm, and non-judgmental
- Never give medical advice
- Never replace a therapist or doctor
- Focus on one clear idea at a time
- Use emoji sparingly — maximum 1 per message

When told about an emotion, provide exactly:
1. A gentle explanation of what the emotion feels like (1 sentence)
2. One simple social tip for this emotion (1 sentence)
3. One regulation suggestion (breathing, movement, etc.) (1 sentence)
"""

# ---------------------------------------------------------------------------
# Per-emotion built-in responses (used by fallback & as Gemini seed context)
# ---------------------------------------------------------------------------
EMOTION_RESPONSES = {
    'happy': {
        'emoji': '😊',
        'explanation': "Feeling happy means your heart feels light and warm inside.",
        'social_tip':  "If you see a friend, you can smile or say something kind to them.",
        'regulation':  "Take a slow breath and enjoy this good feeling — it belongs to you. 😊",
    },
    'sad': {
        'emoji': '😢',
        'explanation': "Feeling sad means something feels heavy or hurts a little inside.",
        'social_tip':  "It is okay to tell a trusted person 'I feel sad today.'",
        'regulation':  "Try hugging something soft, or breathe in slowly for 4 counts then out for 4.",
    },
    'angry': {
        'emoji': '😤',
        'explanation': "Feeling angry means something upset or frustrated you.",
        'social_tip':  "Before speaking, try counting to 5 quietly — it helps words come out calmer.",
        'regulation':  "Squeeze your hands into fists, then open them slowly — repeat 3 times.",
    },
    'fear': {
        'emoji': '😨',
        'explanation': "Feeling scared means your body is telling you something feels unsafe.",
        'social_tip':  "You can move closer to a safe person or place you trust.",
        'regulation':  "Breathe in for 4, hold for 4, breathe out for 4 — this calms your body.",
    },
    'surprise': {
        'emoji': '😲',
        'explanation': "Feeling surprised means something unexpected just happened.",
        'social_tip':  "It is okay to say 'I did not expect that!' — people understand.",
        'regulation':  "Take one big breath, look around slowly, and notice 3 things you can see.",
    },
    'disgust': {
        'emoji': '😕',
        'explanation': "Feeling disgusted means something seems very unpleasant to you.",
        'social_tip':  "You can politely move away from what bothers you — that is okay.",
        'regulation':  "Look at something you like instead, and breathe through your mouth slowly.",
    },
    'neutral': {
        'emoji': '😐',
        'explanation': "You seem calm and balanced right now — that is perfectly fine.",
        'social_tip':  "A calm moment is a great time to ask someone how their day is going.",
        'regulation':  "Keep breathing steadily and take things one small step at a time.",
    },
}


def _format_autism_feedback(emotion: str, confidence: float) -> str:
    """Build a formatted autism-friendly feedback string from built-in responses."""
    data = EMOTION_RESPONSES.get(emotion.lower(), EMOTION_RESPONSES['neutral'])
    conf_pct = int(confidence * 100)
    lines = [
        f"{data['emoji']} You seem to be feeling **{emotion}** right now ({conf_pct}% sure).",
        f"💡 {data['explanation']}",
        f"🤝 {data['social_tip']}",
        f"🌿 {data['regulation']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GeminiClient — live API
# ---------------------------------------------------------------------------
class GeminiClient:
    """
    Wraps google-generativeai to provide autism-friendly emotional support.
    Requires a valid GEMINI_API_KEY in the environment / .env.
    """

    def __init__(self):
        self.is_available = False
        self.model = None
        if not GENAI_AVAILABLE:
            print("❌ google-generativeai not installed. Run: pip install google-generativeai")
            return
        if not _API_KEY:
            print("❌ GEMINI_API_KEY not set. Using fallback client.")
            return
        try:
            genai.configure(api_key=_API_KEY)
            self.model = genai.GenerativeModel(
                model_name=_MODEL,
                system_instruction=AUTISM_SYSTEM_PROMPT,
            )
            # Quick smoke-test (lightweight)
            self.is_available = True
            print(f"✅ Gemini client ready ({_MODEL})")
        except Exception as e:
            print(f"❌ Gemini init failed: {e}")

    # ------------------------------------------------------------------
    def get_response(self, prompt: str) -> str:
        """General chat response."""
        if not self.is_available or self.model is None:
            return GeminiFallbackClient().get_response(prompt)
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=200,
                    temperature=0.7,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            print(f"Gemini get_response error: {e}")
            return GeminiFallbackClient().get_response(prompt)

    def get_autism_feedback(self, emotion: str, confidence: float) -> str:
        """
        Autism-specific structured feedback for a given detected emotion.
        Returns gentle, simple, 3-part text.
        """
        if not self.is_available or self.model is None:
            return _format_autism_feedback(emotion, confidence)

        prompt = (
            f"The detected emotion is: {emotion} (confidence: {int(confidence * 100)}%).\n\n"
            f"Give an autism-friendly response following your guidelines:\n"
            f"1. Gentle explanation\n"
            f"2. Social suggestion\n"
            f"3. Regulation tip\n"
            f"Keep total response under 4 sentences. Use plain language."
        )
        try:
            resp = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=180,
                    temperature=0.6,
                ),
            )
            return resp.text.strip()
        except Exception as e:
            print(f"Gemini autism_feedback error: {e}")
            return _format_autism_feedback(emotion, confidence)


# ---------------------------------------------------------------------------
# GeminiFallbackClient — no API key required
# ---------------------------------------------------------------------------
class GeminiFallbackClient:
    """
    Used when Gemini API key is missing or the SDK is not installed.
    Returns high-quality pre-written autism-friendly responses.
    """

    def __init__(self):
        self.is_available = False  # signals to app.py that it is a fallback

    def get_response(self, prompt: str) -> str:
        """Best-effort keyword-based chat reply."""
        p = prompt.lower()
        if any(w in p for w in ['hello', 'hi ', 'hey']):
            return "Hello! I'm here to help you. How are you feeling today? 😊"
        if any(w in p for w in ['sad', 'cry', 'unhappy', 'depressed']):
            return ("It is okay to feel sad sometimes. Take a slow breath. "
                    "You are not alone — I am right here with you.")
        if any(w in p for w in ['angry', 'frustrated', 'mad']):
            return ("It sounds like something upset you. That is understandable. "
                    "Try squeezing your hands tight, then opening them slowly.")
        if any(w in p for w in ['scared', 'afraid', 'fear', 'worried']):
            return ("Feeling scared is normal. You are safe right now. "
                    "Breathe in for 4 counts, then out for 4 counts.")
        if any(w in p for w in ['happy', 'good', 'great', 'excited']):
            return "That is wonderful! 😊 Enjoy this good feeling — you deserve it."
        return ("Thank you for sharing with me. Remember to breathe slowly "
                "and take things one step at a time. I am here for you.")

    def get_autism_feedback(self, emotion: str, confidence: float) -> str:
        return _format_autism_feedback(emotion, confidence)
