class SessionReport:
    def __init__(self, emotion_timeline):
        """
        Initializes the SessionReport generator.
        
        Args:
            emotion_timeline (EmotionTimeline): Timeline object containing session data.
        """
        self.timeline = emotion_timeline.get_timeline()
        self.report = None
        
    def generate_report(self):
        """
        Calculates session metrics and returns the report.
        
        Returns:
            dict: The session report JSON structure. Example:
            {
                "dominant_emotion": "happy",
                "happy": "40%",
                "neutral": "30%",
                ...
            }
        """
        if not self.timeline:
            return {"error": "No data available to generate report"}
            
        emotion_counts = {}
        total_records = len(self.timeline)
        
        # Count all occurrences
        for entry in self.timeline:
            em = entry["emotion"]
            emotion_counts[em] = emotion_counts.get(em, 0) + 1
            
        # Determine dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
        
        # Calculate percentages
        report = {
            "dominant_emotion": dominant_emotion,
            "total_frames_processed": total_records
        }
        
        for em, count in emotion_counts.items():
            percentage = (count / total_records) * 100
            report[em] = f"{int(percentage)}%"
            
        self.report = report
        return report
