import pyaudio
import numpy as np
import threading
from queue import Queue
from config import Config

class AudioPlayer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.is_playing = False
        self.stream = None
        self.audio_queue = Queue()
        
    def start_playback(self, sample_rate=44100, channels=1):
        """Start audio playback stream"""
        if self.is_playing:
            return False
            
        try:
            self.stream = self.audio.open(
                format=pyaudio.paFloat32,
                channels=channels,
                rate=sample_rate,
                output=True,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.is_playing = True
            print("✅ Audio playback started")
            return True
            
        except Exception as e:
            print(f"❌ Error starting audio playback: {e}")
            return False
    
    def stop_playback(self):
        """Stop audio playback"""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.is_playing = False
        print("✅ Audio playback stopped")
    
    def add_audio_data(self, audio_data):
        """Add audio data to playback queue"""
        if self.is_playing and audio_data is not None:
            self.audio_queue.put(audio_data)
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio playback callback"""
        try:
            if not self.audio_queue.empty():
                audio_data = self.audio_queue.get()
                # Convert to the right format for playback
                if isinstance(audio_data, np.ndarray):
                    audio_data = audio_data.astype(np.float32).tobytes()
                return (audio_data, pyaudio.paContinue)
            else:
                # Return silence if no data
                return (b'\x00' * frame_count * 4, pyaudio.paContinue)  # 4 bytes per float32
                
        except Exception as e:
            print(f"Audio playback error: {e}")
            return (b'\x00' * frame_count * 4, pyaudio.paContinue)
    
    def __del__(self):
        """Cleanup"""
        self.stop_playback()
        self.audio.terminate()

# Global audio player instance
audio_player = AudioPlayer()