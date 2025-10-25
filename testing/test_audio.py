from audio_processing.audio_utils import AudioRecorder
from audio_processing.simple_emotion_detector import SimpleAudioEmotionDetector

def test_audio_system():
    print("Testing audio system...")
    
    try:
        # Initialize components
        recorder = AudioRecorder()
        detector = SimpleAudioEmotionDetector()
        
        # Start recording
        print("Starting audio recording...")
        recorder.start_recording()
        
        # Get a few chunks and analyze
        for i in range(5):
            chunk = recorder.get_audio_chunk()
            if chunk is not None:
                emotion, confidence = detector.real_time_emotion_analysis(chunk)
                print(f"Chunk {i+1}: {emotion} (confidence: {confidence:.2f})")
            else:
                print(f"Chunk {i+1}: No audio data")
        
        # Stop recording
        recorder.stop_recording()
        print("✅ Audio system test completed successfully")
        
    except Exception as e:
        print(f"❌ Audio system test failed: {e}")

if __name__ == "__main__":
    test_audio_system()