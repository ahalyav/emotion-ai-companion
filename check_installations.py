import importlib
import sys
import pkg_resources

def check_module(module_name, package_name=None):
    """Check if a module is installed and get its version"""
    try:
        if package_name is None:
            package_name = module_name
        
        version = pkg_resources.get_distribution(package_name).version
        print(f"✅ {module_name}: {version}")
        return True
    except pkg_resources.DistributionNotFound:
        print(f"❌ {module_name}: Not installed")
        return False
    except Exception as e:
        print(f"⚠️  {module_name}: Error - {e}")
        return False

def check_import(module_name):
    """Check if a module can be imported"""
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name}: Can be imported")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: Import failed - {e}")
        return False

print("=" * 50)
print("Emotion AI Companion - Installation Check")
print("=" * 50)

# Core dependencies
print("\n📦 CORE DEPENDENCIES:")
core_modules = [
    ('flask', 'Flask'),
    ('cv2', 'opencv-python'),
    ('numpy', 'numpy'),
    ('dotenv', 'python-dotenv'),
    ('librosa', 'librosa'),
    ('sounddevice', 'sounddevice'),
    ('soundfile', 'soundfile'),
    ('sklearn', 'scikit-learn'),
    ('requests', 'requests'),
    ('PIL', 'Pillow'),
    ('matplotlib', 'matplotlib'),
]

for module, package in core_modules:
    check_module(module, package)

# Optional dependencies
print("\n🔧 OPTIONAL DEPENDENCIES:")
optional_modules = [
    ('tensorflow', 'tensorflow'),
    ('deepface', 'deepface'),
    ('torch', 'torch'),
]

for module, package in optional_modules:
    check_module(module, package)

# Check custom module imports
print("\n🏗️  CUSTOM MODULES:")
custom_modules = [
    'audio_processing.audio_utils',
    'audio_processing.simple_emotion_detector',
    'video_processing.camera_utils', 
    'video_processing.simple_face_detector',
    'fusion_engine.emotion_fusion',
    'llm_companion.ollama_client',
    'config'
]

for module in custom_modules:
    check_import(module)

# Check Python version
print(f"\n🐍 PYTHON VERSION: {sys.version}")

print("\n" + "=" * 50)
print("Installation Check Complete")
print("=" * 50)