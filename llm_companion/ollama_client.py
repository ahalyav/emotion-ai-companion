import requests
import json
import time
from config import Config

class OllamaClient:
    def __init__(self):
        self.base_url = Config.OLLAMA_BASE_URL
        self.model = Config.OLLAMA_MODEL
        self.is_available = self.check_availability()
        
    def check_availability(self):
        """Check if Ollama is running and available"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                print("✅ Ollama is available")
                return True
            else:
                print("❌ Ollama is not responding properly")
                return False
        except Exception as e:
            print(f"❌ Ollama is not available: {e}")
            return False
    
    def generate_response(self, prompt, emotion_data, max_tokens=150):
        """Generate response based on emotion and context"""
        if not self.is_available:
            return "Ollama is not available. Please make sure Ollama is running."
        
        try:
            # Create emotion-aware prompt
            enhanced_prompt = self.create_emotion_aware_prompt(prompt, emotion_data)
            
            payload = {
                "model": self.model,
                "prompt": enhanced_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": max_tokens
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'No response generated.')
            else:
                return f"Error: {response.status_code} - {response.text}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def create_emotion_aware_prompt(self, user_message, emotion_data):
        """Create a prompt that considers the user's emotional state"""
        current_emotion = emotion_data.get('current_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.0)
        
        emotion_context = {
            'happy': "The user appears to be happy and positive. Respond in an uplifting, cheerful manner that matches their mood.",
            'sad': "The user seems to be feeling sad. Respond with empathy, comfort, and support. Offer gentle encouragement.",
            'anger': "The user appears to be angry or frustrated. Respond calmly and help de-escalate the situation. Be understanding but not confrontational.",
            'surprise': "The user seems surprised. Respond with curiosity and engagement. Help them process whatever surprised them.",
            'fear': "The user appears anxious or fearful. Respond with reassurance, safety, and support. Help them feel more secure.",
            'disgust': "The user seems displeased or disgusted. Respond with understanding and offer alternative perspectives.",
            'neutral': "The user appears neutral. Respond in a balanced, informative, and helpful manner."
        }
        
        emotion_guidance = emotion_context.get(current_emotion, emotion_context['neutral'])
        
        prompt = f"""You are an empathetic AI companion. {emotion_guidance}

Current emotional state detected: {current_emotion} (confidence: {confidence:.2f})

User message: {user_message}

Provide a helpful, appropriate response that considers the user's emotional state:"""
        
        return prompt

# Mock client for when Ollama is not available
class MockOllamaClient:
    def __init__(self):
        self.is_available = False
        
    def generate_response(self, prompt, emotion_data, max_tokens=150):
        emotion = emotion_data.get('current_emotion', 'neutral')
        
        responses = {
            'happy': f"I can tell you're feeling happy! That's wonderful. Regarding your message '{prompt}', I'm here to help keep that positive energy going! 😊",
            'sad': f"I sense you might be feeling down. I'm here for you. About '{prompt}' - let me help you work through this. Remember, every cloud has a silver lining. 🌧️➡️🌈",
            'anger': f"I notice some frustration in your tone. Let's take a deep breath together. About '{prompt}' - I'm here to help resolve this calmly.",
            'surprise': f"You seem surprised! That's interesting. Regarding '{prompt}' - let's explore this together. What surprised you most?",
            'fear': f"I sense some anxiety. You're safe here. About '{prompt}' - let me help you feel more secure and work through this step by step.",
            'neutral': f"Thanks for your message about '{prompt}'. I'm here to help you with that in a calm and focused way."
        }
        
        return responses.get(emotion, f"Thanks for sharing: '{prompt}'. I'm here to help you with that.")