from audio_processing.audio_utils import AudioRecorder
from audio_processing.simple_emotion_detector import SimpleAudioEmotionDetector
import time

def test_audio_with_detector():
    print("Testing audio system with emotion detection...")
    
    try:
        # Initialize components
        recorder = AudioRecorder()
        detector = SimpleAudioEmotionDetector()
        
        # Start recording
        print("Starting audio recording...")
        if not recorder.start_recording():
            print("❌ Failed to start recording")
            return
            
        # Wait a moment for audio to start flowing
        time.sleep(0.5)
        
        # Get and analyze multiple chunks
        successful_chunks = 0
        for i in range(10):
            chunk = recorder.get_audio_chunk()
            if chunk is not None:
                emotion, confidence = detector.real_time_emotion_analysis(chunk)
                print(f"Chunk {i+1}: {emotion} (confidence: {confidence:.2f})")
                successful_chunks += 1
            else:
                print(f"Chunk {i+1}: No audio data")
            
            time.sleep(0.1)  # Small delay between chunks
        
        # Stop recording
        recorder.stop_recording()
        
        print(f"\n📊 Results: {successful_chunks}/10 chunks successfully processed")
        if successful_chunks > 0:
            print("✅ Audio system is working correctly!")
        else:
            print("❌ Audio system still has issues")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    test_audio_with_detector()