import sounddevice as sd
import numpy as np
import time
import threading
from queue import Queue
from config import Config

class AudioRecorder:
    def __init__(self):
        self.sample_rate = Config.AUDIO_SAMPLE_RATE
        self.chunk_size = Config.AUDIO_CHUNK_SIZE
        self.channels = Config.AUDIO_CHANNELS
        self.is_recording = False
        self.audio_stream = None
        self.audio_queue = Queue()
        self.recording_start_time = None
        self.total_recorded_chunks = 0
        self.current_session_id = None
        
        # Audio device configuration
        self.device_index = self._find_best_input_device()
        print(f"🎤 Using audio device: {self.device_index}")
        
    def _find_best_input_device(self):
        """Find the best available input device"""
        try:
            devices = sd.query_devices()
            preferred_devices = [
                "Microphone Array (Intel® Smart",  # Your primary device
                "Microphone Array",  # Generic microphone array
                "Microphone",  # Generic microphone
            ]
            
            for i, device in enumerate(devices):
                # Check if device has input channels and matches preferred names
                if (device['max_input_channels'] > 0 and 
                    any(pref in device['name'] for pref in preferred_devices)):
                    print(f"✅ Found input device {i}: {device['name']}")
                    return i
            
            # Fallback to default input device
            default_input = sd.default.device[0]
            if default_input >= 0 and devices[default_input]['max_input_channels'] > 0:
                print(f"⚠️ Using default input device {default_input}: {devices[default_input]['name']}")
                return default_input
                
            # Last resort: use any device with input channels
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    print(f"⚠️ Using fallback device {i}: {device['name']}")
                    return i
                    
            print("❌ No suitable input devices found!")
            return None
            
        except Exception as e:
            print(f"❌ Error finding audio devices: {e}")
            return None
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback function for audio stream"""
        if status:
            print(f"Audio stream status: {status}")
        
        if self.is_recording:
            try:
                # Add audio data to queue
                audio_chunk = indata.copy()
                
                # Basic audio validation
                if np.any(np.isnan(audio_chunk)) or np.any(np.isinf(audio_chunk)):
                    print("⚠️ Invalid audio data detected")
                    return
                
                self.audio_queue.put(audio_chunk)
                self.total_recorded_chunks += 1
                
            except Exception as e:
                print(f"Error in audio callback: {e}")
    
    def start_recording(self):
        """Start audio recording"""
        if self.is_recording:
            print("⚠️ Audio recording already in progress")
            return True
            
        if self.device_index is None:
            print("❌ No audio device available")
            return False
            
        try:
            # Reset queue and counters
            while not self.audio_queue.empty():
                self.audio_queue.get()
            
            self.total_recorded_chunks = 0
            self.recording_start_time = time.time()
            self.current_session_id = f"audio_{int(time.time() * 1000)}"
            
            print(f"🔊 Starting audio recording...")
            print(f"   Sample rate: {self.sample_rate} Hz")
            print(f"   Chunk size: {self.chunk_size}")
            print(f"   Channels: {self.channels}")
            print(f"   Device: {self.device_index}")
            
            # Start audio stream with explicit device selection
            self.audio_stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.chunk_size,
                channels=self.channels,
                callback=self._audio_callback,
                dtype='float32',
                device=self.device_index
            )
            
            self.audio_stream.start()
            self.is_recording = True
            
            # Wait a moment for stream to stabilize
            time.sleep(0.1)
            
            print(f"✅ Audio recording started - Session: {self.current_session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error starting audio recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """Stop audio recording"""
        if not self.is_recording:
            print("⚠️ No active recording to stop")
            return True
            
        try:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            
            self.is_recording = False
            duration = time.time() - self.recording_start_time if self.recording_start_time else 0
            
            print(f"✅ Audio recording stopped. Duration: {duration:.2f}s, Chunks: {self.total_recorded_chunks}")
            return True
            
        except Exception as e:
            print(f"❌ Error stopping audio recording: {e}")
            return False
    
    def get_audio_chunk(self):
        """Get the latest audio chunk from the queue"""
        if not self.is_recording:
            return None
            
        try:
            # Get all available chunks and return the latest one
            latest_chunk = None
            chunk_count = 0
            
            while not self.audio_queue.empty():
                latest_chunk = self.audio_queue.get()
                chunk_count += 1
            
            # If we had multiple chunks, print a warning (shouldn't happen often)
            if chunk_count > 1:
                print(f"⚠️ Skipped {chunk_count - 1} old audio chunks")
            
            return latest_chunk
            
        except Exception as e:
            print(f"Error getting audio chunk: {e}")
            return None
    
    def get_audio_stats(self):
        """Get statistics about the current audio stream"""
        if not self.is_recording:
            return {"status": "Not recording"}
        
        try:
            latest_chunk = self.get_audio_chunk()
            if latest_chunk is not None:
                audio_data = np.array(latest_chunk).flatten()
                return {
                    "status": "Recording",
                    "rms": float(np.sqrt(np.mean(audio_data**2))),
                    "max_amplitude": float(np.max(np.abs(audio_data))),
                    "chunks_processed": self.total_recorded_chunks
                }
            else:
                return {
                    "status": "Recording (no data)",
                    "chunks_processed": self.total_recorded_chunks
                }
        except Exception as e:
            return {"status": f"Error: {str(e)}"}
    
    def get_session_info(self):
        """Get current session information"""
        duration = time.time() - self.recording_start_time if self.recording_start_time else 0
        
        return {
            'session_id': self.current_session_id,
            'duration': duration,
            'chunks_recorded': self.total_recorded_chunks,
            'is_recording': self.is_recording,
            'sample_rate': self.sample_rate,
            'device_index': self.device_index
        }

# Test function
def test_audio_recorder():
    """Test the audio recorder"""
    print("Testing AudioRecorder...")
    
    recorder = AudioRecorder()
    
    # Start recording
    if recorder.start_recording():
        print("Recording started successfully")
        
        # Collect some audio chunks
        chunks_received = 0
        for i in range(10):
            time.sleep(0.1)  # Wait for audio data
            chunk = recorder.get_audio_chunk()
            if chunk is not None:
                chunks_received += 1
                print(f"Chunk {i+1}: Received audio data, shape: {chunk.shape}")
            else:
                print(f"Chunk {i+1}: No audio data")
        
        print(f"\n📊 Results: {chunks_received}/10 chunks received")
        
        # Stop recording
        recorder.stop_recording()
        
        if chunks_received > 0:
            print("✅ Audio recorder test PASSED")
        else:
            print("❌ Audio recorder test FAILED - no audio data received")
    else:
        print("❌ Failed to start recording")

if __name__ == "__main__":
    test_audio_recorder()