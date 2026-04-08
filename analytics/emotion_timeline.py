class EmotionTimeline:
    def __init__(self):
        """
        Initializes the EmotionTimeline array.
        """
        self.timeline = []
        
    def log_emotion(self, timestamp, emotion, confidence):
        """
        Logs a new emotion into the timeline.
        
        Args:
            timestamp (float): The current timestamp.
            emotion (str): The smoothed emotion detected.
            confidence (float): The confidence value.
        """
        if emotion is None or confidence is None:
            return
            
        entry = {
            "time": timestamp,
            "emotion": emotion,
            "confidence": confidence
        }
        
        self.timeline.append(entry)
        
    def get_timeline(self):
        """
        Returns the entire logged session timeline.
        
        Returns:
            list: List of dictionaries representing the session timeline.
        """
        return self.timeline
