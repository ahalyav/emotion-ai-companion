import numpy as np
from flask import Flask, render_template, jsonify, Response, request, send_file
import threading
import time
import json
import sys
import cv2
import base64
import os
import glob
import tempfile
from datetime import datetime
from collections import deque
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
    try:
        from audio_processing.audio_utils import AudioRecorder
        from audio_processing.simple_emotion_detector import SimpleAudioEmotionDetector
        AUDIO_AVAILABLE = True
        print("✅ Basic audio processing available")
    except ImportError as e2:
        print(f"❌ Audio modules not available: {e2}")

# Import video processing
VIDEO_AVAILABLE = False
face_detector = None
camera = None

try:
    from video_processing.camera_utils import Camera
    camera = Camera()
    if camera.initialize():
        print("✅ Camera initialized successfully")
        try:
            from video_processing.simple_face_detector import SimpleFaceDetector
            face_detector = SimpleFaceDetector()
            VIDEO_AVAILABLE = True
            print("✅ Enhanced face detector loaded")
        except Exception as e:
            print(f"❌ Face detector failed: {e}")
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
    print("✅ Fusion engine (Bayesian) available")
except Exception as e:
    print(f"❌ Fusion engine not available: {e}")

# Import new modules
try:
    from video_processing.emotion_model import FaceEmotionRecognizer
    from fusion.emotion_smoothing import EmotionSmoothing
    from analytics.emotion_timeline import EmotionTimeline
    from ai.autism_feedback import AutismFeedback
    from analytics.session_report import SessionReport
    
    face_emotion_recognizer = FaceEmotionRecognizer()
    emotion_smoother = EmotionSmoothing(history_size=5)
    emotion_timeline = EmotionTimeline()
    autism_feedback = AutismFeedback()
    
    print("✅ System Upgrades Modules Loaded")
except Exception as e:
    print(f"❌ System Upgrades Modules failed to load: {e}")
    face_emotion_recognizer = None
    emotion_smoother = None
    emotion_timeline = None
    autism_feedback = None

# ---------------------------------------------------------------------------
# Import LLM — Gemini (primary) → Ollama (optional) → Fallback
# ---------------------------------------------------------------------------
LLM_AVAILABLE = False
GEMINI_AVAILABLE = False
ollama_client = None
gemini_client = None

try:
    from llm_companion.gemini_client import GeminiClient, GeminiFallbackClient
    _gc = GeminiClient()
    if _gc.is_available:
        gemini_client = _gc
        GEMINI_AVAILABLE = True
        LLM_AVAILABLE = True
        print("✅ Gemini LLM available")
    else:
        gemini_client = GeminiFallbackClient()
        print("✅ Using Gemini fallback client (no API key)")
except Exception as e:
    print(f"❌ Gemini import failed: {e}")
    from llm_companion.gemini_client import GeminiFallbackClient
    gemini_client = GeminiFallbackClient()

try:
    from llm_companion.ollama_client import OllamaClient, MockOllamaClient
    ollama_client = OllamaClient()
    if ollama_client.is_available:
        LLM_AVAILABLE = True
        print("✅ Ollama LLM also available (optional fallback)")
    else:
        ollama_client = MockOllamaClient()
except Exception as e:
    print(f"❌ Ollama not available: {e}")
    try:
        from llm_companion.ollama_client import MockOllamaClient
        ollama_client = MockOllamaClient()
    except Exception:
        pass

app = Flask(__name__)
app.config.from_object(Config)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024   # 20 MB upload limit
ALLOWED_AUDIO_EXT = {'wav', 'mp3', 'ogg', 'flac'}
ALLOWED_VIDEO_EXT = {'mp4', 'avi', 'mov', 'webm'}


def _allowed_file(filename, extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in extensions

# -----------------------------------------------------------------------
# Global state
# -----------------------------------------------------------------------
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
    'gemini_available': GEMINI_AVAILABLE,
    'processing_active': False,
    'audio_chunks_processed': 0,
    'video_frames_processed': 0,
    'recording_start_time': None,
    'recording_duration': 0,
    # Raw 7-class scores for radar chart
    'fused_scores': {e: 0.0 for e in ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']},
    'audio_scores': {},
    'face_scores': {},
}

# Circular buffer: last 60 timestamped emotion snapshots
HISTORY_DEPTH = getattr(Config, 'EMOTION_HISTORY_DEPTH', 60)
emotion_history_buffer = deque(maxlen=HISTORY_DEPTH)

# Session storage
recorded_sessions = {
    'audio_sessions': [],
    'video_sessions': [],
    'combined_sessions': []
}

# Chat history (rolling window)
CHAT_HISTORY_MAX = getattr(Config, 'CHAT_HISTORY_MAX', 10)
chat_history = deque(maxlen=CHAT_HISTORY_MAX * 2)   # user+assistant pairs

# Audio components
audio_recorder = None
audio_detector = None

if AUDIO_AVAILABLE:
    try:
        audio_recorder = AudioRecorder()
        try:
            audio_detector = MFCCEmotionDetector()
            print("✅ Advanced MFCC emotion detector initialized")
        except Exception:
            from audio_processing.simple_emotion_detector import SimpleAudioEmotionDetector
            audio_detector = SimpleAudioEmotionDetector()
            print("✅ Basic audio emotion detector initialized")
        print("✅ Audio components initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing audio components: {e}")
        AUDIO_AVAILABLE = False

stop_camera_thread = threading.Event()


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def clean_emotion_data(data):
    """Convert numpy arrays to lists for JSON serialization."""
    cleaned = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            cleaned[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            cleaned[key] = [
                item.tolist() if isinstance(item, np.ndarray) else item
                for item in value
            ]
        else:
            cleaned[key] = value
    return cleaned


def _snapshot_emotion():
    """Push a timestamped emotion snapshot into the history buffer."""
    snap = {
        'timestamp': time.time(),
        'emotion': emotion_data.get('current_emotion', 'neutral'),
        'confidence': emotion_data.get('confidence', 0.0),
        'audio_emotion': emotion_data.get('audio_emotion', 'neutral'),
        'face_emotion': emotion_data.get('face_emotion', 'neutral'),
    }
    emotion_history_buffer.append(snap)


# -----------------------------------------------------------------------
# Background threads
# -----------------------------------------------------------------------

def background_audio_processing():
    global emotion_data
    if not AUDIO_AVAILABLE or audio_recorder is None:
        return
    while True:
        try:
            if audio_recorder.is_recording:
                audio_chunk = audio_recorder.get_audio_chunk()
                if audio_chunk is not None:
                    try:
                        emotion, confidence = audio_detector.real_time_emotion_analysis(audio_chunk)
                        emotion_data['audio_emotion'] = emotion
                        emotion_data['audio_confidence'] = confidence
                        emotion_data['audio_chunks_processed'] += 1
                        emotion_data['processing_active'] = True
                        # Store raw scores if available
                        if hasattr(audio_detector, 'get_raw_scores'):
                            emotion_data['audio_scores'] = audio_detector.get_raw_scores()
                    except Exception as e:
                        print(f"Audio analysis error: {e}")
                        emotion_data['audio_emotion'] = 'neutral'
                        emotion_data['audio_confidence'] = 0.5
            time.sleep(0.1)
        except Exception as e:
            print(f"Error in audio processing thread: {e}")
            time.sleep(1)


def background_video_processing():
    global emotion_data
    if not VIDEO_AVAILABLE or camera is None or face_detector is None:
        return
    while not stop_camera_thread.is_set():
        try:
            if camera.is_recording:
                frame = camera.get_frame()
                if frame is not None:
                    # 1. Base Face Detection
                    base_emotion, base_confidence, faces = face_detector.detect_emotion(frame)
                    
                    if len(faces) > 0 and face_emotion_recognizer and emotion_smoother and emotion_timeline:
                        # Find largest face by area
                        largest_face = max(faces, key=lambda f: f[2] * f[3])
                        x, y, w, h = largest_face
                        
                        # Apply safe padding before cropping
                        x1 = max(0, x - 30)
                        y1 = max(0, y - 30)
                        x2 = min(frame.shape[1], x + w + 30)
                        y2 = min(frame.shape[0], y + h + 30)
                        
                        face_crop = frame[y1:y2, x1:x2]
                        
                        # 2. Advanced Vision Transformer Emotion Model
                        hf_result = face_emotion_recognizer.predict_emotion(face_crop)
                        raw_emotion = hf_result.get("emotion", base_emotion)
                        raw_confidence = hf_result.get("confidence", base_confidence)
                    else:
                        raw_emotion = base_emotion
                        raw_confidence = base_confidence

                    # 3. Temporal Emotion Smoothing
                    if emotion_smoother:
                        stable_result = emotion_smoother.smooth_prediction(raw_emotion, raw_confidence)
                        emotion = stable_result["emotion"]
                        confidence = stable_result["confidence"]
                    else:
                        emotion = raw_emotion
                        confidence = raw_confidence
                        
                    # 4. Emotion Timeline Logger
                    if emotion_timeline:
                        emotion_timeline.log_emotion(time.time(), emotion, confidence)
                        
                    emotion_data['face_emotion'] = emotion
                    emotion_data['face_confidence'] = confidence
                    emotion_data['face_detected'] = len(faces) > 0
                    # Store raw scores if available
                    if hasattr(face_detector, 'get_raw_scores'):
                        emotion_data['face_scores'] = face_detector.get_raw_scores()

                    emotion_data['face_coordinates'] = []
                    for face in faces:
                        if hasattr(face, 'tolist'):
                            emotion_data['face_coordinates'].append(face.tolist())
                        else:
                            emotion_data['face_coordinates'].append(list(face))

                    emotion_data['video_frames_processed'] += 1
                    emotion_data['processing_active'] = True

                    # Draw bounding boxes
                    frame_bgr = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
                    for (x, y, w, h) in faces:
                        color = (0, 255, 120)
                        cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), color, 2)
                        label = f"{emotion} {confidence * 100:.0f}%"
                        cv2.putText(frame_bgr, label,
                                    (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                    cv2.putText(frame_bgr, f"Frames: {emotion_data['video_frames_processed']}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

                    _, buffer = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    emotion_data['video_feed'] = base64.b64encode(buffer).decode('utf-8')

            time.sleep(0.033)
        except Exception as e:
            print(f"Error in video processing: {e}")
            time.sleep(1)


def background_fusion_processing():
    global emotion_data
    while True:
        try:
            if FUSION_AVAILABLE and fusion_engine:
                audio_input = {
                    'emotion': emotion_data.get('audio_emotion', 'neutral'),
                    'confidence': emotion_data.get('audio_confidence', 0.0)
                }
                face_input = {
                    'emotion': emotion_data.get('face_emotion', 'neutral'),
                    'confidence': emotion_data.get('face_confidence', 0.0)
                }
                result = fusion_engine.get_fusion_analysis(
                    audio_input, face_input,
                    audio_scores=emotion_data.get('audio_scores'),
                    face_scores=emotion_data.get('face_scores'),
                )
                emotion_data['current_emotion'] = result['fused_emotion']
                emotion_data['confidence'] = result['fused_confidence']
                emotion_data['modality_agreement'] = result['modality_agreement']
                if 'fused_scores' in result:
                    emotion_data['fused_scores'] = result['fused_scores']
            else:
                if AUDIO_AVAILABLE:
                    emotion_data['current_emotion'] = emotion_data.get('audio_emotion', 'neutral')
                    emotion_data['confidence'] = emotion_data.get('audio_confidence', 0.0)
                elif VIDEO_AVAILABLE:
                    emotion_data['current_emotion'] = emotion_data.get('face_emotion', 'neutral')
                    emotion_data['confidence'] = emotion_data.get('face_confidence', 0.0)

            # Update recording duration
            if emotion_data.get('recording_start_time'):
                emotion_data['recording_duration'] = time.time() - emotion_data['recording_start_time']

            # Push snapshot to history buffer
            _snapshot_emotion()

            time.sleep(0.5)
        except Exception as e:
            print(f"Error in fusion processing: {e}")
            time.sleep(1)


def load_saved_sessions():
    try:
        sessions_path = Config.SESSION_SAVE_PATH
        if not os.path.exists(sessions_path):
            os.makedirs(sessions_path)
            return
        for meta_file in glob.glob(f"{sessions_path}/*_audio_meta.json"):
            try:
                with open(meta_file) as f:
                    recorded_sessions['audio_sessions'].append(json.load(f))
            except Exception as e:
                print(f"Error loading {meta_file}: {e}")
        for meta_file in glob.glob(f"{sessions_path}/*_video_meta.json"):
            try:
                with open(meta_file) as f:
                    recorded_sessions['video_sessions'].append(json.load(f))
            except Exception as e:
                print(f"Error loading {meta_file}: {e}")
        print(f"✅ Loaded {len(recorded_sessions['audio_sessions'])} audio, "
              f"{len(recorded_sessions['video_sessions'])} video sessions")
    except Exception as e:
        print(f"❌ Error loading saved sessions: {e}")


# -----------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------

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
    cleaned = clean_emotion_data(emotion_data)
    return jsonify(cleaned)


@app.route('/api/emotion_scores')
def get_emotion_scores():
    """Return 7-class raw probability scores for all modalities (radar chart)."""
    return jsonify({
        'fused_scores': emotion_data.get('fused_scores', {}),
        'audio_scores': emotion_data.get('audio_scores', {}),
        'face_scores': emotion_data.get('face_scores', {}),
        'timestamp': time.time(),
    })


@app.route('/api/emotion_history')
def get_emotion_history():
    """Return the last N timestamped emotion snapshots (for timeline chart)."""
    return jsonify(list(emotion_history_buffer))


@app.route('/api/system_status')
def get_system_status():
    return jsonify({
        'audio_available': AUDIO_AVAILABLE,
        'video_available': VIDEO_AVAILABLE,
        'fusion_available': FUSION_AVAILABLE,
        'llm_available': LLM_AVAILABLE,
        'gemini_available': GEMINI_AVAILABLE,
        'audio_recording': audio_recorder.is_recording if AUDIO_AVAILABLE and audio_recorder else False,
        'video_recording': camera.is_recording if VIDEO_AVAILABLE and camera else False,
        'system_ready': True,
        'timestamp': time.time()
    })


@app.route('/api/process_status')
def get_process_status():
    return jsonify({
        'video_frames_processed': emotion_data.get('video_frames_processed', 0),
        'audio_chunks_processed': emotion_data.get('audio_chunks_processed', 0),
        'face_detected': emotion_data.get('face_detected', False),
        'audio_processing': 'Active' if AUDIO_AVAILABLE and audio_recorder and audio_recorder.is_recording else 'Idle',
        'video_processing': 'Active' if VIDEO_AVAILABLE and camera and camera.is_recording else 'Idle',
        'fusion_processing': 'Active',
        'current_emotion': emotion_data.get('current_emotion', 'neutral')
    })


@app.route('/api/video_feed')
def get_video_feed():
    if emotion_data.get('video_feed'):
        return jsonify({'frame': emotion_data['video_feed']})
    return jsonify({'frame': None, 'status': 'No video feed available'})


@app.route('/api/recording_status')
def get_recording_status():
    audio_session_info = audio_recorder.get_session_info() if AUDIO_AVAILABLE and audio_recorder else {}
    video_session_info = camera.get_session_info() if VIDEO_AVAILABLE and camera else {}
    return jsonify({
        'audio_recording': audio_recorder.is_recording if AUDIO_AVAILABLE and audio_recorder else False,
        'video_recording': camera.is_recording if VIDEO_AVAILABLE and camera else False,
        'audio_available': AUDIO_AVAILABLE,
        'video_available': VIDEO_AVAILABLE,
        'total_audio_chunks': audio_recorder.total_recorded_chunks if AUDIO_AVAILABLE and audio_recorder else 0,
        'processing_active': emotion_data.get('processing_active', False),
        'recording_duration': emotion_data.get('recording_duration', 0),
        'audio_session': audio_session_info,
        'video_session': video_session_info
    })


@app.route('/api/sessions')
def get_sessions():
    return jsonify(recorded_sessions)


@app.route('/api/start_video')
def start_video():
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
    if not VIDEO_AVAILABLE or not camera:
        return jsonify({'success': False, 'error': 'Video system not available'})
    try:
        camera.stop_recording()
        return jsonify({'success': True, 'message': 'Video recording stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/start_audio')
def start_audio():
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
    if not AUDIO_AVAILABLE or not audio_recorder:
        return jsonify({'success': False, 'error': 'Audio system not available'})
    try:
        audio_recorder.stop_recording()
        return jsonify({'success': True, 'message': 'Audio recording stopped'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/start_all')
def start_all():
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
    return jsonify({
        'success': True,
        'emotion': emotion_data.get('current_emotion', 'neutral'),
        'confidence': emotion_data.get('confidence', 0.5)
    })


@app.route('/api/emotion_timeline')
def get_emotion_timeline():
    """Return the entire session's emotional timeline."""
    if emotion_timeline:
        return jsonify({
            'success': True, 
            'timeline': emotion_timeline.get_timeline()
        })
    return jsonify({'success': False, 'error': 'Timeline module not loaded'})


@app.route('/api/get_ai_feedback')
def get_ai_feedback():
    try:
        current_emotion = emotion_data.get('current_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.5)

        if LLM_AVAILABLE and ollama_client:
            prompt = (
                f"The user appears to be feeling {current_emotion} with "
                f"{confidence:.1%} confidence. Provide a warm, supportive, and "
                f"empathetic response in 2-3 sentences."
            )
            response = ollama_client.get_response(prompt)
            return jsonify({'success': True, 'ai_feedback': response, 'emotion_used': current_emotion})

        # Enhanced fallback using AutismFeedback system
        if autism_feedback:
            fallback = autism_feedback.generate_supportive_feedback(current_emotion)
            return jsonify({'success': True, 'ai_feedback': fallback, 'emotion_used': current_emotion, 'note': 'Autism-friendly fallback response'})

        # If all else fails
        return jsonify({'success': True, 'ai_feedback': "I'm here to listen and support you. How are you feeling today?", 'emotion_used': current_emotion})

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'ai_feedback': "I'm here to listen and support you. How are you feeling today?",
            'note': 'Error occurred, using default response'
        })


@app.route('/api/session_report')
def get_session_report():
    if emotion_timeline:
        report_generator = SessionReport(emotion_timeline)
        report = report_generator.generate_report()
        return jsonify({'success': True, 'report': report})
    return jsonify({'success': False, 'error': 'Session report not available'})


@app.route('/api/chat', methods=['POST'])
def chat():
    """Conversational endpoint: user sends a message, AI responds in context."""
    try:
        data = request.get_json(force=True)
        user_msg = data.get('message', '').strip()
        if not user_msg:
            return jsonify({'success': False, 'error': 'Empty message'})

        current_emotion = emotion_data.get('current_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.0)

        # Build history context string
        history_lines = []
        for msg in chat_history:
            history_lines.append(f"{msg['role'].capitalize()}: {msg['content']}")
        history_ctx = '\n'.join(history_lines[-6:])  # last 3 pairs

        system_context = (
            f"You are a warm AI companion. The user is currently feeling "
            f"{current_emotion} (confidence {confidence:.0%}). "
            f"Respond empathetically and helpfully in 2-4 sentences."
        )
        prompt = (
            f"{system_context}\n\nConversation so far:\n{history_ctx}\n\n"
            f"User: {user_msg}\nAssistant:"
        ) if history_ctx else (
            f"{system_context}\n\nUser: {user_msg}\nAssistant:"
        )

        if LLM_AVAILABLE and ollama_client:
            ai_reply = ollama_client.get_response(prompt)
        else:
            # Mock fallback
            ai_reply = _mock_chat_reply(user_msg, current_emotion)

        # Update chat history
        chat_history.append({'role': 'user', 'content': user_msg})
        chat_history.append({'role': 'assistant', 'content': ai_reply})

        return jsonify({'success': True, 'response': ai_reply, 'emotion_context': current_emotion})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'response': 'Sorry, I had trouble responding just now.'})


def _mock_chat_reply(msg, emotion):
    """Simple mock replies keyed on keywords when Ollama is unavailable."""
    msg_l = msg.lower()
    if any(w in msg_l for w in ['hello', 'hi', 'hey']):
        return f"Hello! I can see you're feeling {emotion} right now. How can I support you today?"
    if any(w in msg_l for w in ['sad', 'unhappy', 'depressed']):
        return "I'm sorry to hear that. It's okay to feel this way. I'm here to listen — what's on your mind?"
    if any(w in msg_l for w in ['happy', 'good', 'great', 'excited']):
        return "That's wonderful to hear! 😊 Your positive energy is contagious. What's been making you happy?"
    if any(w in msg_l for w in ['help', 'advice', 'what should']):
        return f"Based on your current emotional state ({emotion}), I'd suggest taking a moment to breathe and ground yourself. Would you like to talk through what's on your mind?"
    return (f"Thank you for sharing that. Given that you're feeling {emotion}, "
            f"I want you to know that I'm here to support you. What else would you like to talk about?")


@app.route('/api/save_audio_session', methods=['POST'])
def save_audio_session():
    if not AUDIO_AVAILABLE or not audio_recorder:
        return jsonify({'success': False, 'error': 'Audio system not available'})
    try:
        session_id = f"audio_{int(time.time() * 1000)}"
        session_data = {
            'session_id': session_id,
            'type': 'audio',
            'duration': emotion_data.get('recording_duration', 0),
            'emotion': emotion_data.get('audio_emotion', 'neutral'),
            'confidence': emotion_data.get('audio_confidence', 0.0),
            'timestamp': datetime.now().isoformat(),
            'chunks_processed': emotion_data.get('audio_chunks_processed', 0)
        }
        recorded_sessions['audio_sessions'].append(session_data)
        if len(recorded_sessions['audio_sessions']) > Config.MAX_SESSIONS:
            recorded_sessions['audio_sessions'] = recorded_sessions['audio_sessions'][-Config.MAX_SESSIONS:]
        return jsonify({'success': True, 'session': session_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/save_video_session', methods=['POST'])
def save_video_session():
    if not VIDEO_AVAILABLE or not camera:
        return jsonify({'success': False, 'error': 'Video system not available'})
    try:
        session_id = f"video_{int(time.time() * 1000)}"
        session_data = {
            'session_id': session_id,
            'type': 'video',
            'duration': emotion_data.get('recording_duration', 0),
            'emotion': emotion_data.get('face_emotion', 'neutral'),
            'confidence': emotion_data.get('face_confidence', 0.0),
            'timestamp': datetime.now().isoformat(),
            'frames_processed': emotion_data.get('video_frames_processed', 0)
        }
        recorded_sessions['video_sessions'].append(session_data)
        if len(recorded_sessions['video_sessions']) > Config.MAX_SESSIONS:
            recorded_sessions['video_sessions'] = recorded_sessions['video_sessions'][-Config.MAX_SESSIONS:]
        return jsonify({'success': True, 'session': session_data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/test_ollama')
def test_ollama():
    try:
        if LLM_AVAILABLE and ollama_client:
            resp = ollama_client.get_response("Hello! Please respond with 'AI Companion is working!'")
            return jsonify({'success': True, 'response': resp, 'ollama_status': 'Connected'})
        return jsonify({'success': False, 'ollama_status': 'Not available', 'message': 'Using mock client'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'ollama_status': 'Connection failed'})


@app.route('/api/start_audio_playback')
def start_audio_playback():
    return jsonify({'success': True, 'message': 'Audio playback ready'})


@app.route('/api/stop_audio_playback')
def stop_audio_playback():
    return jsonify({'success': True, 'message': 'Audio playback stopped'})


@app.route('/api/toggle_monitor')
def toggle_monitor():
    return jsonify({'success': True, 'message': 'Use the browser audio controls to toggle monitoring', 'monitoring': True})


# ---------------------------------------------------------------------------
# Gemini Autism Feedback Endpoint
# ---------------------------------------------------------------------------

@app.route('/api/gemini_feedback')
def get_gemini_feedback():
    """Return autism-friendly Gemini-generated feedback for the current fused emotion."""
    try:
        current_emotion = emotion_data.get('current_emotion', 'neutral')
        confidence = emotion_data.get('confidence', 0.5)
        if gemini_client is not None:
            feedback_text = gemini_client.get_autism_feedback(current_emotion, confidence)
        else:
            feedback_text = f"You seem to be feeling {current_emotion} right now. Take a slow breath and know that you are safe. 😊"
        return jsonify({
            'success': True,
            'emotion': current_emotion,
            'confidence': round(confidence, 3),
            'feedback': feedback_text,
            'gemini_available': GEMINI_AVAILABLE,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'feedback': 'I am here for you. Take a deep breath. 🌿'})


# ---------------------------------------------------------------------------
# File Upload — Audio Analysis
# ---------------------------------------------------------------------------

@app.route('/api/upload_audio', methods=['POST'])
def upload_audio():
    """Analyse an uploaded audio file and return the detected emotion."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    if not _allowed_file(file.filename, ALLOWED_AUDIO_EXT):
        return jsonify({'success': False, 'error': f'Unsupported audio format. Allowed: {ALLOWED_AUDIO_EXT}'})

    tmp_path = None
    try:
        suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        # Load audio with librosa
        import librosa
        audio_data, sr = librosa.load(tmp_path, sr=None, mono=True)
        if sr != Config.AUDIO_SAMPLE_RATE:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=Config.AUDIO_SAMPLE_RATE)

        # Run MFCC detector
        if AUDIO_AVAILABLE and audio_detector is not None:
            emotion, confidence = audio_detector.detect_emotion(audio_data)
            raw_scores = audio_detector.get_raw_scores() if hasattr(audio_detector, 'get_raw_scores') else {}
        else:
            from audio_processing.emotion_detector import MFCCEmotionDetector
            _det = MFCCEmotionDetector()
            emotion, confidence = _det.detect_emotion(audio_data)
            raw_scores = _det.get_raw_scores()

        # Gemini feedback
        feedback = ''
        if gemini_client is not None:
            feedback = gemini_client.get_autism_feedback(emotion, confidence)

        return jsonify({
            'success': True,
            'emotion': emotion,
            'confidence': round(float(confidence), 3),
            'scores': raw_scores,
            'feedback': feedback,
            'filename': file.filename,
            'duration_seconds': round(len(audio_data) / Config.AUDIO_SAMPLE_RATE, 2),
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ---------------------------------------------------------------------------
# File Upload — Video Analysis
# ---------------------------------------------------------------------------

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Analyse an uploaded video file and return the aggregated emotion."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part in request'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    if not _allowed_file(file.filename, ALLOWED_VIDEO_EXT):
        return jsonify({'success': False, 'error': f'Unsupported video format. Allowed: {ALLOWED_VIDEO_EXT}'})

    tmp_path = None
    try:
        suffix = '.' + file.filename.rsplit('.', 1)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            file.save(tmp_path)

        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            return jsonify({'success': False, 'error': 'Could not open video file'})

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        duration = total_frames / fps if fps > 0 else 0

        # Sample up to 15 evenly-spaced frames
        MAX_FRAMES = 15
        frame_indices = (
            list(range(0, total_frames, max(1, total_frames // MAX_FRAMES)))[:MAX_FRAMES]
            if total_frames > 0 else [0]
        )

        if VIDEO_AVAILABLE and face_detector is not None:
            _fd = face_detector
        else:
            from video_processing.simple_face_detector import SimpleFaceDetector
            _fd = SimpleFaceDetector()

        emotion_counts = {}
        frames_analysed = 0
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame_bgr = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            emotion, conf, _ = _fd.detect_emotion(frame_rgb)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            frames_analysed += 1
        cap.release()

        if not emotion_counts:
            return jsonify({'success': False, 'error': 'No frames could be analysed (no faces found)'})

        dominant = max(emotion_counts, key=emotion_counts.get)
        dom_conf = emotion_counts[dominant] / frames_analysed if frames_analysed else 0.5
        raw_scores = _fd.get_raw_scores()

        # Gemini feedback
        feedback = ''
        if gemini_client is not None:
            feedback = gemini_client.get_autism_feedback(dominant, dom_conf)

        return jsonify({
            'success': True,
            'emotion': dominant,
            'confidence': round(float(dom_conf), 3),
            'scores': raw_scores,
            'emotion_distribution': emotion_counts,
            'frames_analysed': frames_analysed,
            'duration_seconds': round(duration, 2),
            'feedback': feedback,
            'filename': file.filename,
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Emotion AI Companion — Enhanced Edition")
    print("=" * 55)
    print(f"  Audio System : {'✅ Available' if AUDIO_AVAILABLE else '❌ Not Available'}")
    print(f"  Video System : {'✅ Available' if VIDEO_AVAILABLE else '❌ Not Available'}")
    print(f"  Fusion Engine: {'✅ Bayesian Fusion' if FUSION_AVAILABLE else '❌ Not Available'}")
    print(f"  LLM Companion: {'✅ Ollama' if LLM_AVAILABLE else '🤖 Mock Mode'}")

    load_saved_sessions()

    if AUDIO_AVAILABLE:
        threading.Thread(target=background_audio_processing, daemon=True).start()
        print("✅ Background audio processing started")

    if VIDEO_AVAILABLE:
        threading.Thread(target=background_video_processing, daemon=True).start()
        print("✅ Background video processing started")

    threading.Thread(target=background_fusion_processing, daemon=True).start()
    print("✅ Background fusion (Bayesian) processing started")

    print("\n🚀 Flask server starting → http://localhost:5000")
    print("=" * 55)

    try:
        app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
    finally:
        stop_camera_thread.set()
        if AUDIO_AVAILABLE and audio_recorder:
            audio_recorder.stop_recording()
        if VIDEO_AVAILABLE and camera:
            camera.stop_camera()
        print("✅ Cleanup complete")