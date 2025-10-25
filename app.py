import numpy as np  # Add this if missing
from flask import Flask, render_template, jsonify, Response, request, send_file
import threading
import time
import json
import sys
import cv2
import base64
import os
import glob
from datetime import datetime
from config import Config

print("Initializing Emotion AI Companion with Session Recording...")

# Import audio processing with enhanced MFCC detector
AUDIO_AVAILABLE = False
try:
    from audio_processing.audio_utils import AudioRecorder
    from audio_processing.emotion_detector import MFCCEmotionDetector
    AUDIO_AVAILABLE = True
    print("✅ Advanced audio processing available (MFCC)")
except ImportError as e:
    print(f"❌ Advanced audio modules not available: {e}")
    # Fallback to simple detector
    try:
        from audio_processing.audio_utils import AudioRecorder
        from audio_processing.simple_emotion_detector import SimpleAudioEmotionDetector
        AUDIO_AVAILABLE = True
        print("✅ Basic audio processing available")
    except ImportError as e2:
        print(f"❌ Audio modules not available: {e2}")

# Import video processing with comprehensive fallbacks
VIDEO_AVAILABLE = False
face_detector = None
camera = None

try:
    # Try to import and initialize camera
    from video_processing.camera_utils import Camera
    camera = Camera()
    if camera.initialize():
        print("✅ Camera initialized successfully")
        
        # Try simple face detector first (no TensorFlow dependency)
        try:
            from video_processing.simple_face_detector import SimpleFaceDetector
            face_detector = SimpleFaceDetector()
            VIDEO_AVAILABLE = True
            print("✅ Simple face detector loaded")
        except Exception as e:
            print(f"❌ Simple face detector failed: {e}")
            
    else:
        print("❌ Camera initialization failed")
        
except Exception as e:
    print(f"❌ Video system initialization failed: {e}")

# Import fusion engine
FUSION_AVAILABLE = False
try:
    from fusion_engine.emotion_fusion import EmotionFusion
    fusion_engine = EmotionFusion()
    FUSION_AVAILABLE = True
    print("✅ Fusion engine available")
except Exception as e:
    print(f"❌ Fusion engine not available: {e}")

# Import LLM companion
LLM_AVAILABLE = False
try:
    from llm_companion.ollama_client import OllamaClient, MockOllamaClient
    # Try real Ollama client first, fallback to mock
    ollama_client = OllamaClient()
    if ollama_client.is_available:
        LLM_AVAILABLE = True
        print("✅ Ollama LLM available")
    else:
        ollama_client = MockOllamaClient()
        print("✅ Using mock LLM (Ollama not available)")
except Exception as e:
    print(f"❌ LLM modules not available: {e}")
    # Create mock client as fallback
    from llm_companion.ollama_client import MockOllamaClient
    ollama_client = MockOllamaClient()
    print("✅ Using mock LLM (fallback)")

app = Flask(__name__)
app.config.from_object(Config)

# Global variables
emotion_data = {
    'current_emotion': 'neutral',
    'audio_emotion': 'neutral',
    'face_emotion': 'neutral', 
    'confidence': 0.0,
    'audio_confidence': 0.0,
    'face_confidence': 0.0,
    'system_status': 'Ready to Start Recording',
    'face_detected': False,
    'face_coordinates': [],
    'video_feed': None,
    'modality_agreement': True,
    'llm_available': LLM_AVAILABLE,
    'processing_active': False,
    'audio_chunks_processed': 0,
    'video_frames_processed': 0,
    'recording_start_time': None,
    'recording_duration': 0
}

# Session storage
recorded_sessions = {
    'audio_sessions': [],
    'video_sessions': [],
    'combined_sessions': []
}

# Chat history
chat_history = []

# Initialize audio components
audio_recorder = None
audio_detector = None

if AUDIO_AVAILABLE:
    try:
        audio_recorder = AudioRecorder()
        # Try advanced detector first
        try:
            audio_detector = MFCCEmotionDetector()
            print("✅ Advanced MFCC emotion detector initialized")
        except:
            from audio_processing.simple_emotion_detector import SimpleAudioEmotionDetector
            audio_detector = SimpleAudioEmotionDetector()
            print("✅ Basic audio emotion detector initialized")
        print("✅ Audio components initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing audio components: {e}")
        AUDIO_AVAILABLE = False

# Thread control
stop_camera_thread = threading.Event()

def clean_emotion_data(data):
    """Convert numpy arrays to lists for JSON serialization"""
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            # Convert numpy arrays to lists
            cleaned[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            # Handle lists/tuples that might contain numpy arrays
            cleaned_list = []
            for item in value:
                if isinstance(item, np.ndarray):
                    cleaned_list.append(item.tolist())
                else:
                    cleaned_list.append(item)
            cleaned[key] = cleaned_list
        else:
            cleaned[key] = value
    return cleaned

def background_audio_processing():
    """Background thread for audio emotion detection"""
    global emotion_data
    
    if not AUDIO_AVAILABLE or audio_recorder is None:
        return
        
    while True:
        try:
            if audio_recorder.is_recording:
                audio_chunk = audio_recorder.get_audio_chunk()
                if audio_chunk is not None:
                    # Add error handling for the audio chunk
                    try:
                        emotion, confidence = audio_detector.real_time_emotion_analysis(audio_chunk)
                        emotion_data['audio_emotion'] = emotion
                        emotion_data['audio_confidence'] = confidence
                        emotion_data['audio_chunks_processed'] += 1
                        emotion_data['processing_active'] = True
                    except Exception as e:
                        print(f"Audio analysis error: {e}")
                        # Set neutral emotion on error
                        emotion_data['audio_emotion'] = 'neutral'
                        emotion_data['audio_confidence'] = 0.5
                    
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            time.sleep(1)

def background_video_processing():
    """Background thread for video emotion detection"""
    global emotion_data
    
    if not VIDEO_AVAILABLE or camera is None or face_detector is None:
        return
        
    while not stop_camera_thread.is_set():
        try:
            if camera.is_recording:
                frame = camera.get_frame()
                if frame is not None:
                    emotion, confidence, faces = face_detector.detect_emotion(frame)
                    emotion_data['face_emotion'] = emotion
                    emotion_data['face_confidence'] = confidence
                    emotion_data['face_detected'] = len(faces) > 0
                    
                    # Convert numpy arrays to lists for JSON serialization
                    emotion_data['face_coordinates'] = []
                    for face in faces:
                        if hasattr(face, 'tolist'):
                            emotion_data['face_coordinates'].append(face.tolist())
                        else:
                            emotion_data['face_coordinates'].append(list(face))
                    
                    emotion_data['video_frames_processed'] += 1
                    emotion_data['processing_active'] = True
                    
                    # Update video feed with face detection
                    frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                    for (x, y, w, h) in faces:
                        color = (0, 255, 0)  # Green for detected face
                        cv2.rectangle(frame_bgr, (x, y), (x+w, y+h), color, 3)
                        cv2.putText(frame_bgr, f"{emotion} ({confidence:.1f})", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Add timestamp and status
                    cv2.putText(frame_bgr, f"Frames: {emotion_data['video_frames_processed']}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    emotion_data['video_feed'] = base64.b64encode(buffer).decode('utf-8')
                    
            time.sleep(0.033)
        except Exception as e:
            print(f"Error in video processing: {e}")
            time.sleep(1)

def background_fusion_processing():
    """Background thread for emotion fusion"""
    global emotion_data
    
    while True:
        try:
            if FUSION_AVAILABLE and fusion_engine:
                audio_data = {
                    'emotion': emotion_data.get('audio_emotion', 'neutral'),
                    'confidence': emotion_data.get('audio_confidence', 0.0)
                }
                
                face_data = {
                    'emotion': emotion_data.get('face_emotion', 'neutral'),
                    'confidence': emotion_data.get('face_confidence', 0.0)
                }
                
                fusion_result = fusion_engine.get_fusion_analysis(audio_data, face_data)
                emotion_data['current_emotion'] = fusion_result['fused_emotion']
                emotion_data['confidence'] = fusion_result['fused_confidence']
                emotion_data['modality_agreement'] = fusion_result['modality_agreement']
            else:
                # Fallback: use audio emotion if available, otherwise face emotion
                if AUDIO_AVAILABLE:
                    emotion_data['current_emotion'] = emotion_data.get('audio_emotion', 'neutral')
                    emotion_data['confidence'] = emotion_data.get('audio_confidence', 0.0)
                elif VIDEO_AVAILABLE:
                    emotion_data['current_emotion'] = emotion_data.get('face_emotion', 'neutral')
                    emotion_data['confidence'] = emotion_data.get('face_confidence', 0.0)
            
            # Update recording duration
            if emotion_data.get('recording_start_time'):
                emotion_data['recording_duration'] = time.time() - emotion_data['recording_start_time']
            
            time.sleep(0.5)
        except Exception as e:
            print(f"Error in fusion processing: {e}")
            time.sleep(1)

def load_saved_sessions():
    """Load previously saved sessions from disk"""
    try:
        sessions_path = Config.SESSION_SAVE_PATH
        if not os.path.exists(sessions_path):
            os.makedirs(sessions_path)
            return
        
        # Load audio sessions
        audio_files = glob.glob(f"{sessions_path}/*_audio_meta.json")
        for meta_file in audio_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    recorded_sessions['audio_sessions'].append(metadata)
            except Exception as e:
                print(f"Error loading audio session {meta_file}: {e}")
        
        # Load video sessions
        video_files = glob.glob(f"{sessions_path}/*_video_meta.json")
        for meta_file in video_files:
            try:
                with open(meta_file, 'r') as f:
                    metadata = json.load(f)
                    recorded_sessions['video_sessions'].append(metadata)
            except Exception as e:
                print(f"Error loading video session {meta_file}: {e}")
        
        print(f"✅ Loaded {len(recorded_sessions['audio_sessions'])} audio sessions and {len(recorded_sessions['video_sessions'])} video sessions")
        
    except Exception as e:
        print(f"❌ Error loading saved sessions: {e}")

@app.route('/')
def index():
    status_info = {
        'audio': 'Available' if AUDIO_AVAILABLE else 'Not Available',
        'video': 'Available' if VIDEO_AVAILABLE else 'Not Available',
        'fusion': 'Available' if FUSION_AVAILABLE else 'Not Available',
        'llm': 'Available' if LLM_AVAILABLE else 'Mock Mode',
        'detector_type': 'Multi-Modal Emotion Detection',
        'system': 'Ready to use',
        'audio_detector': 'MFCC Advanced' if 'MFCC' in str(type(audio_detector)) else 'Basic'
    }
    return render_template('index.html', status=status_info)

@app.route('/api/emotion')
def get_emotion():
    # Clean the data before converting to JSON
    cleaned_data = clean_emotion_data(emotion_data)
    return jsonify(cleaned_data)

@app.route('/api/system_status')
def get_system_status():
    """Get comprehensive system status"""
    status = {
        'audio_available': AUDIO_AVAILABLE,
        'video_available': VIDEO_AVAILABLE,
        'fusion_available': FUSION_AVAILABLE,
        'llm_available': LLM_AVAILABLE,
        'audio_recording': audio_recorder.is_recording if AUDIO_AVAILABLE and audio_recorder else False,
        'video_recording': camera.is_recording if VIDEO_AVAILABLE and camera else False,
        'system_ready': True,
        'timestamp': time.time()
    }
    return jsonify(status)

@app.route('/api/process_status')
def get_process_status():
    """Get processing status"""
    status = {
        'video_frames_processed': emotion_data.get('video_frames_processed', 0),
        'audio_chunks_processed': emotion_data.get('audio_chunks_processed', 0),
        'face_detected': emotion_data.get('face_detected', False),
        'audio_processing': 'Active' if AUDIO_AVAILABLE and audio_recorder and audio_recorder.is_recording else 'Idle',
        'video_processing': 'Active' if VIDEO_AVAILABLE and camera and camera.is_recording else 'Idle',
        'fusion_processing': 'Active',
        'current_emotion': emotion_data.get('current_emotion', 'neutral')
    }
    return jsonify(status)

@app.route('/api/video_feed')
def get_video_feed():
    """Get current video frame"""
    if emotion_data.get('video_feed'):
        return jsonify({'frame': emotion_data['video_feed']})
    else:
        # Return a placeholder or black frame
        return jsonify({'frame': None, 'status': 'No video feed available'})

@app.route('/api/recording_status')
def get_recording_status():
    """Get current recording status"""
    audio_session_info = audio_recorder.get_session_info() if AUDIO_AVAILABLE and audio_recorder else {}
    video_session_info = camera.get_session_info() if VIDEO_AVAILABLE and camera else {}
    
    status = {
        'audio_recording': audio_recorder.is_recording if AUDIO_AVAILABLE and audio_recorder else False,
        'video_recording': camera.is_recording if VIDEO_AVAILABLE and camera else False,
        'audio_available': AUDIO_AVAILABLE,
        'video_available': VIDEO_AVAILABLE,
        'total_audio_chunks': audio_recorder.total_recorded_chunks if AUDIO_AVAILABLE and audio_recorder else 0,
        'processing_active': emotion_data.get('processing_active', False),
        'recording_duration': emotion_data.get('recording_duration', 0),
        'audio_session': audio_session_info,
        'video_session': video_session_info
    }
    return jsonify(status)

@app.route('/api/sessions')
def get_sessions():
    """Get all recorded sessions"""
    return jsonify(recorded_sessions)

@app.route('/api/start_video')
def start_video():
    """Start video recording"""
    if not VIDEO_AVAILABLE or not camera:
        return jsonify({'success': False, 'error': 'Video system not available'})
    
    try:
        camera.start_recording()
        emotion_data['recording_start_time'] = time.time()
        return jsonify({'success': True, 'message': 'Video recording started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_video')
def stop_video():
    """Stop video recording"""
    if not VIDEO_AVAILABLE or not camera:
        return jsonify({'success': False, 'error': 'Video system not available'})
    
    try:
        camera.stop_recording()
        return jsonify({'success': True, 'message': 'Video recording stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_audio')
def start_audio():
    """Start audio recording"""
    if not AUDIO_AVAILABLE or not audio_recorder:
        return jsonify({'success': False, 'error': 'Audio system not available'})
    
    try:
        audio_recorder.start_recording()
        emotion_data['recording_start_time'] = time.time()
        return jsonify({'success': True, 'message': 'Audio recording started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stop_audio')
def stop_audio():
    """Stop audio recording"""
    if not AUDIO_AVAILABLE or not audio_recorder:
        return jsonify({'success': False, 'error': 'Audio system not available'})
    
    try:
        audio_recorder.stop_recording()
        return jsonify({'success': True, 'message': 'Audio recording stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/start_all')
def start_all():
    """Start both audio and video recording"""
    results = {}
    
    if AUDIO_AVAILABLE and audio_recorder:
        audio_recorder.start_recording()
        results['audio'] = 'started'
    
    if VIDEO_AVAILABLE and camera:
        camera.start_recording()
        results['video'] = 'started'
    
    emotion_data['recording_start_time'] = time.time()
    return jsonify({'success': True, 'results': results})

@app.route('/api/stop_all')
def stop_all():
    """Stop both audio and video recording"""
    results = {}
    
    if AUDIO_AVAILABLE and audio_recorder:
        audio_recorder.stop_recording()
        results['audio'] = 'stopped'
    
    if VIDEO_AVAILABLE and camera:
        camera.stop_recording()
        results['video'] = 'stopped'
    
    return jsonify({'success': True, 'results': results})

@app.route('/api/record_sample')
def record_sample():
    """Record a short sample"""
    # This would record a short sample in a real implementation
    return jsonify({
        'success': True, 
        'emotion': emotion_data.get('current_emotion', 'neutral'),
        'confidence': emotion_data.get('confidence', 0.5)
    })

@app.route('/api/get_ai_feedback')
def get_ai_feedback():
    """Get AI feedback based on current emotion"""
    try:
        print(f"🔍 Debug: Starting AI feedback request...")
        print(f"🔍 Debug: LLM_AVAILABLE = {LLM_AVAILABLE}")
        print(f"🔍 Debug: ollama_client type = {type(ollama_client)}")
        
        if LLM_AVAILABLE and ollama_client:
            current_emotion = emotion_data.get('current_emotion', 'neutral')
            confidence = emotion_data.get('confidence', 0.5)
            
            prompt = f"The user appears to be feeling {current_emotion} with {confidence:.1%} confidence. Provide a supportive and empathetic response in 2-3 sentences."
            
            print(f"🔍 Debug: Prompt = {prompt}")
            
            response = ollama_client.get_response(prompt)
            print(f"🔍 Debug: Response received = {response}")
            
            return jsonify({
                'success': True,
                'ai_feedback': response,
                'emotion_used': current_emotion
            })
        else:
            print(f"🔍 Debug: Using fallback mock response")
            # Enhanced fallback responses based on emotion
            emotion_responses = {
                'happy': "I can see you're feeling happy! That's wonderful! 😊 Keep spreading that positive energy - it looks great on you!",
                'sad': "I notice you're feeling a bit down. Remember that every cloud has a silver lining. I'm here to listen if you want to talk. 💙",
                'angry': "I sense some frustration. Taking a deep breath can help. Would you like to talk about what's bothering you?",
                'surprise': "You seem surprised! Life is full of unexpected moments. Embrace the wonder and curiosity! ✨",
                'fear': "It's okay to feel apprehensive sometimes. Remember that you're stronger than you think. What's on your mind?",
                'neutral': "I notice you're feeling balanced and neutral right now. How can I help you today?",
                'disgust': "I sense some discomfort. Sometimes stepping back and reframing the situation can help gain perspective."
            }
            
            current_emotion = emotion_data.get('current_emotion', 'neutral')
            fallback_response = emotion_responses.get(current_emotion, emotion_responses['neutral'])
            
            return jsonify({
                'success': True,
                'ai_feedback': fallback_response,
                'emotion_used': current_emotion,
                'note': 'Using enhanced fallback response'
            })
            
    except Exception as e:
        print(f"❌ Error in get_ai_feedback: {str(e)}")
        import traceback
        print(f"❌ Full traceback: {traceback.format_exc()}")
        
        return jsonify({
            'success': False,
            'error': str(e),
            'ai_feedback': "I'm here to listen and support you. How are you feeling today?",
            'note': 'Error occurred, using default response'
        })
    
    
@app.route('/api/save_audio_session', methods=['POST'])
def save_audio_session():
    """Save current audio session"""
    if not AUDIO_AVAILABLE or not audio_recorder:
        return jsonify({'success': False, 'error': 'Audio system not available'})
    
    try:
        # Get current emotion data for metadata
        emotion_metadata = {
            'final_emotion': emotion_data.get('audio_emotion', 'neutral'),
            'final_confidence': emotion_data.get('audio_confidence', 0.0),
            'total_chunks': emotion_data.get('audio_chunks_processed', 0)
        }
        
        # In a real implementation, you would save the actual audio data
        # For now, we'll create a session record
        session_id = f"audio_{int(time.time() * 1000)}"
        session_data = {
            'session_id': session_id,
            'type': 'audio',
            'duration': emotion_data.get('recording_duration', 0),
            'emotion': emotion_metadata['final_emotion'],
            'confidence': emotion_metadata['final_confidence'],
            'timestamp': datetime.now().isoformat(),
            'chunks_processed': emotion_metadata['total_chunks']
        }
        
        recorded_sessions['audio_sessions'].append(session_data)
        
        # Keep only recent sessions
        if len(recorded_sessions['audio_sessions']) > Config.MAX_SESSIONS:
            recorded_sessions['audio_sessions'] = recorded_sessions['audio_sessions'][-Config.MAX_SESSIONS:]
        
        return jsonify({'success': True, 'session': session_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/test_ollama')
def test_ollama():
    """Test Ollama connection"""
    try:
        if LLM_AVAILABLE and ollama_client:
            test_prompt = "Hello! Please respond with 'AI Companion is working!' if you can hear me."
            response = ollama_client.get_response(test_prompt)
            return jsonify({
                'success': True,
                'response': response,
                'ollama_status': 'Connected and working'
            })
        else:
            return jsonify({
                'success': False,
                'ollama_status': 'Not available',
                'message': 'Using mock client'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'ollama_status': 'Connection failed'
        })

@app.route('/api/save_video_session', methods=['POST'])
def save_video_session():
    """Save current video session"""
    if not VIDEO_AVAILABLE or not camera:
        return jsonify({'success': False, 'error': 'Video system not available'})
    
    try:
        # Get current emotion data for metadata
        emotion_metadata = {
            'final_emotion': emotion_data.get('face_emotion', 'neutral'),
            'final_confidence': emotion_data.get('face_confidence', 0.0),
            'total_frames': emotion_data.get('video_frames_processed', 0)
        }
        
        # In a real implementation, you would save the actual video data
        # For now, we'll create a session record
        session_id = f"video_{int(time.time() * 1000)}"
        session_data = {
            'session_id': session_id,
            'type': 'video',
            'duration': emotion_data.get('recording_duration', 0),
            'emotion': emotion_metadata['final_emotion'],
            'confidence': emotion_metadata['final_confidence'],
            'timestamp': datetime.now().isoformat(),
            'frames_processed': emotion_metadata['total_frames']
        }
        
        recorded_sessions['video_sessions'].append(session_data)
        
        # Keep only recent sessions
        if len(recorded_sessions['video_sessions']) > Config.MAX_SESSIONS:
            recorded_sessions['video_sessions'] = recorded_sessions['video_sessions'][-Config.MAX_SESSIONS:]
        
        return jsonify({'success': True, 'session': session_data})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# Audio Playback Endpoints
@app.route('/api/start_audio_playback')
def start_audio_playback():
    """Start real-time audio playback"""
    return jsonify({
        'success': True, 
        'message': 'Audio playback ready - Use browser controls to monitor',
        'note': 'Real-time monitoring is handled by the browser'
    })

@app.route('/api/stop_audio_playback')
def stop_audio_playback():
    """Stop real-time audio playback"""
    return jsonify({'success': True, 'message': 'Audio playback stopped'})

@app.route('/api/toggle_monitor')
def toggle_monitor():
    """Toggle audio monitoring"""
    return jsonify({
        'success': True, 
        'message': 'Use the browser audio controls to toggle monitoring',
        'monitoring': True
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("Emotion AI Companion - Enhanced with Session Recording")
    print("="*50)
    print(f"Audio System: {'✅ Available' if AUDIO_AVAILABLE else '❌ Not Available'}")
    print(f"Video System: {'✅ Available' if VIDEO_AVAILABLE else '❌ Not Available'}")
    print(f"Fusion Engine: {'✅ Available' if FUSION_AVAILABLE else '❌ Not Available'}")
    print(f"LLM Companion: {'✅ Ollama Available' if LLM_AVAILABLE else '🤖 Mock Mode'}")
    
    if AUDIO_AVAILABLE:
        detector_type = 'MFCC Advanced' if 'MFCC' in str(type(audio_detector)) else 'Basic'
        print(f"Audio Detector: {detector_type}")
    
    if VIDEO_AVAILABLE:
        detector_type = 'Enhanced' if 'FacialEmotionDetector' in str(type(face_detector)) else 'Basic/Mock'
        print(f"Video Detector: {detector_type}")
    
    if LLM_AVAILABLE:
        print(f"Ollama Model: {Config.OLLAMA_MODEL}")
    
    # Load saved sessions
    load_saved_sessions()
    
    # Start background processing threads
    if AUDIO_AVAILABLE:
        audio_thread = threading.Thread(target=background_audio_processing, daemon=True)
        audio_thread.start()
        print("✅ Background audio processing started")
    
    if VIDEO_AVAILABLE:
        video_thread = threading.Thread(target=background_video_processing, daemon=True)
        video_thread.start()
        print("✅ Background video processing started")
    
    fusion_thread = threading.Thread(target=background_fusion_processing, daemon=True)
    fusion_thread.start()
    print("✅ Background fusion processing started")
    
    print("🚀 Starting Flask server...")
    print("💡 Access the application at: http://localhost:5000")
    print("💬 LLM Companion is ready for interaction!")
    print("\n🎯 Enhanced Features:")
    print("  📹 Video recording with session management")
    print("  🎵 Audio recording with session management") 
    print("  🔊 Real-time audio playback monitoring")
    print("  💾 Save and load recording sessions")
    print("  📊 Session metadata and emotion tracking")
    print("  🎭 Real-time multi-modal emotion fusion")
    print("  🤖 AI feedback based on emotional state")
    
    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        # Cleanup
        stop_camera_thread.set()
        if AUDIO_AVAILABLE and audio_recorder:
            audio_recorder.stop_recording()
        if VIDEO_AVAILABLE and camera:
            camera.stop_camera()
        print("✅ Cleanup completed")