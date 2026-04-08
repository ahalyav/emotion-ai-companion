import os
from transformers import pipeline
import cv2
import numpy as np
from PIL import Image

print("Loading generic emotion HuggingFace model 'trpakov/vit-face-expression' (this happens only once)...")
# Initialize the pipeline globally so we load the model once
emotion_classifier = pipeline(
    "image-classification",
    model="trpakov/vit-face-expression",
    device=-1,  # Use CPU for stable non-blocking inference <200ms
    framework="pt" # Force it to PyTorch so it doesn't crash on Tensorflow
)
print("✅ Emotion classifier pipeline initialized.")

class FaceEmotionRecognizer:
    def __init__(self):
        """
        Initializes the FaceEmotionRecognizer.
        The model is loaded globally via pipeline to prevent repeated loading overhead.
        """
        pass

    def predict_emotion(self, face_image):
        """
        Predicts the emotion of a given cropped face image using the ViT model.
        
        Args:
            face_image (numpy.ndarray): Cropped face image (BGR from OpenCV).
            
        Returns:
            dict: Format: {"emotion": "happy", "confidence": 0.83}
        """
        try:
            if face_image is None or face_image.size == 0:
                return {"emotion": "neutral", "confidence": 0.0}

            # Convert BGR to RGB for PIL Image processing
            face_img_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # The transformers pipeline handles resizing (224x224) and normalization
            pil_img = Image.fromarray(face_img_rgb)
            
            # Inference using the global pipeline
            predictions = emotion_classifier(pil_img)
            
            if predictions and len(predictions) > 0:
                # Get the highest scored emotion
                top_prediction = predictions[0]
                label = top_prediction['label'].lower()
                score = top_prediction['score']
                
                # Standardize labels to our expected domain: 
                # ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
                # trpakov/vit-face-expression outputs standard 7-class emotions
                
                return {
                    "emotion": label,
                    "confidence": float(score)
                }
            
            return {"emotion": "neutral", "confidence": 0.0}
            
        except Exception as e:
            print(f"Error in FaceEmotionRecognizer prediction: {e}")
            return {"emotion": "neutral", "confidence": 0.0}
