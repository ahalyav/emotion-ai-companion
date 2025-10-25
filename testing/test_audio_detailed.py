import sounddevice as sd
import numpy as np
import time

def list_audio_devices():
    """List all available audio devices"""
    print("Available audio devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"{i}: {device['name']} (Input channels: {device['max_input_channels']})")

def test_audio_input():
    """Test if we can capture audio from the default device"""
    print("\nTesting audio input...")
    
    try:
        # Test recording a short sample
        duration = 3  # seconds
        sample_rate = 16000
        
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()  # Wait until recording is finished
        
        print(f"✅ Recording successful! Shape: {audio_data.shape}")
        print(f"Audio stats - Min: {np.min(audio_data):.4f}, Max: {np.max(audio_data):.4f}, Mean: {np.mean(audio_data):.4f}")
        
        # Check if we actually captured sound
        rms = np.sqrt(np.mean(audio_data**2))
        if rms < 0.001:
            print("⚠️ Warning: Very low audio level detected. Check your microphone!")
        else:
            print(f"✅ Good audio level detected: {rms:.4f}")
            
        return True
        
    except Exception as e:
        print(f"❌ Audio input test failed: {e}")
        return False

def test_default_device():
    """Test the default audio device"""
    print("\nTesting default audio device...")
    try:
        default_device = sd.default.device
        print(f"Default input device: {default_device[0]}")
        
        device_info = sd.query_devices(default_device[0])
        print(f"Device info: {device_info['name']}")
        print(f"Max input channels: {device_info['max_input_channels']}")
        print(f"Default sample rate: {device_info['default_samplerate']}")
        
        return True
    except Exception as e:
        print(f"❌ Default device test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Audio System Diagnostic Test")
    print("=" * 50)
    
    list_audio_devices()
    test_default_device()
    test_audio_input()