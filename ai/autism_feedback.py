class AutismFeedback:
    def __init__(self):
        """
        Initializes the AutismFeedback generator.
        """
        pass
        
    def generate_supportive_feedback(self, emotion):
        """
        Generates supportive, autism-friendly feedback based on the given emotion.
        
        Args:
            emotion (str): The current stable emotion.
            
        Returns:
            str: Predefined supportive text message.
        """
        emotion = emotion.lower()
        
        rules = {
            "happy": "You seem happy! That's wonderful.",
            "sad": "It's okay to feel sad. Try taking a deep breath.",
            "angry": "You may be feeling frustrated. Let's pause and relax.",
            "fear": "Something might be worrying you. You're safe here.",
            "neutral": "I'm here with you. Tell me how you're feeling.",
            # Handling other possible Standard 7-class emotion outputs
            "disgust": "I sense some discomfort. Let's step back and take a moment.",
            "surprise": "Wow, what a surprise! It's okay, let's take it all in."
        }
        
        return rules.get(emotion, rules["neutral"])
