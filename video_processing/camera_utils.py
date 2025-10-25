import cv2
import time

class Camera:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_recording = False
        self.frame_count = 0
        self.start_time = None
        
    def initialize(self):
        """Initialize camera"""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            return self.cap.isOpened()
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def get_frame(self):
        """Get current frame from camera"""
        if not self.cap or not self.cap.isOpened():
            return None
            
        try:
            ret, frame = self.cap.read()
            if ret:
                self.frame_count += 1
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb
            return None
        except Exception as e:
            print(f"Error getting frame: {e}")
            return None
    
    def start_recording(self):
        """Start recording"""
        if not self.cap:
            if not self.initialize():
                return False
                
        self.is_recording = True
        self.start_time = time.time()
        self.frame_count = 0
        return True
    
    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        return True
    
    def stop_camera(self):
        """Release camera resources"""
        if self.cap:
            self.cap.release()
        self.is_recording = False
    
    def get_session_info(self):
        """Get recording session information"""
        return {
            'frames_captured': self.frame_count,
            'recording_time': time.time() - self.start_time if self.start_time else 0,
            'is_recording': self.is_recording
        }