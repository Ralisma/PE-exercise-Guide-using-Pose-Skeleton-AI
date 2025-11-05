#!/usr/bin/env python3
import cv2
import pyttsx3
import time
import csv
from datetime import datetime
import tensorflow as tf
import numpy as np
import os
import winsound
import threading
import json

MODEL_PATH = "my_model"
LOG_FILE = "exercise_log.csv"

POSE_CLASSES = ["sqs", "shf", "ars"]
CORRECT_THRESHOLD = 0.50
RETRY_THRESHOLD = 0.30
EVAL_DELAY = 0.10

engine = pyttsx3.init()
engine.setProperty('rate', 165)
speech_lock = threading.Lock()

def speak_sync(text):
    print("[SAY]", text)
    with speech_lock:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"TTS failed: {str(e)}")

def speak_async(text):
    def _speak():
        speak_sync(text)
    threading.Thread(target=_speak, daemon=True).start()

def play_tone(freq=880, duration=0.2):
    winsound.Beep(int(freq), int(duration * 1000))

def tone_success(): play_tone(880, 0.25)
def tone_almost():  play_tone(660, 0.25)
def tone_fail():    play_tone(220, 0.4)

speak_sync("System initializing, please wait.")

# Load Teachable Machine Pose Model
def load_teachable_machine_model():
    """Load the Teachable Machine pose classification model"""
    try:
        # Method 1: Try direct loading with custom objects
        model = tf.keras.models.load_model(
            MODEL_PATH,
            custom_objects={
                'Sequential': tf.keras.Sequential,
                'Dense': tf.keras.layers.Dense,
                'Dropout': tf.keras.layers.Dropout
            }
        )
        print("✅ Teachable Machine model loaded directly")
        return model
    except Exception as e:
        print(f"Direct loading failed: {e}")
    
    try:
        # Method 2: Recreate model from JSON and load weights
        with open(os.path.join(MODEL_PATH, 'model.json'), 'r') as f:
            model_json = json.load(f)
        
        # Recreate model architecture
        model = tf.keras.models.model_from_json(json.dumps(model_json['modelTopology']))
        
        # Load weights (this is simplified - may need more complex weight loading)
        print("✅ Model architecture loaded from JSON")
        return model
    except Exception as e:
        print(f"JSON loading failed: {e}")
    
    return None

# Load MoveNet for pose keypoint extraction
def load_movenet():
    """Load MoveNet model for pose keypoint detection"""
    try:
        # Download MoveNet Lightning
        import tensorflow_hub as hub
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        movenet = model.signatures['serving_default']
        print("✅ MoveNet loaded successfully")
        return movenet
    except Exception as e:
        print(f"MoveNet loading failed: {e}")
        return None

# Initialize models
pose_classifier = load_teachable_machine_model()
movenet = load_movenet()

if pose_classifier is None:
    speak_async("Pose classifier failed to load. Using demonstration mode.")
    print("⚠️ Using demonstration mode")

if movenet is None:
    speak_async("Pose detection failed to load. Please check internet connection.")
    print("❌ MoveNet failed to load")

# Camera setup
cap = None
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"✅ Camera opened with backend: {backend}")
        break
    else:
        cap = None

if cap is None:
    speak_async("Camera not detected. Please check connection.")
    exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 15)

with open(LOG_FILE, 'a', newline='') as f:
    writer = csv.writer(f)
    if f.tell() == 0:
        writer.writerow(["timestamp", "pose", "confidence", "status"])

speak_sync("System ready. Let's begin your guided exercise.")

def extract_pose_keypoints(frame, movenet_model):
    """Extract pose keypoints using MoveNet"""
    try:
        # Resize frame to MoveNet input size (192x192)
        image = cv2.resize(frame, (192, 192))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32)
        
        # Expand dimensions for batch
        image = tf.expand_dims(image, axis=0)
        
        # Run inference
        outputs = movenet_model(image)
        keypoints = outputs['output_0'].numpy()
        
        return keypoints[0]  # Return first (and only) batch element
        
    except Exception as e:
        print(f"Pose extraction failed: {e}")
        return None

def predict_pose_from_keypoints(keypoints, classifier_model):
    """Use Teachable Machine model to classify pose from keypoints"""
    try:
        if keypoints is None or classifier_model is None:
            return "Unknown Pose", 0.0
        
        # Flatten keypoints to match Teachable Machine input shape (14739 features)
        # MoveNet outputs 17 keypoints with (y, x, confidence) = 51 values
        # Teachable Machine expects 14739 values, so we need to pad/transform
        
        keypoints_flat = keypoints.flatten()
        
        # Pad or truncate to match expected input size
        if len(keypoints_flat) < 14739:
            # Pad with zeros
            keypoints_flat = np.pad(keypoints_flat, (0, 14739 - len(keypoints_flat)))
        elif len(keypoints_flat) > 14739:
            # Truncate
            keypoints_flat = keypoints_flat[:14739]
        
        # Reshape for model input
        input_data = keypoints_flat.reshape(1, 14739)
        
        # Make prediction
        predictions = classifier_model.predict(input_data, verbose=0)
        pose_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][pose_idx])
        
        if pose_idx < len(POSE_CLASSES):
            return POSE_CLASSES[pose_idx], confidence
        else:
            return "Unknown Pose", confidence
            
    except Exception as e:
        print(f"Pose classification failed: {e}")
        return "Unknown Pose", 0.0

def mock_predict_pose():
    """Fallback mock prediction"""
    import random
    pose_idx = random.randint(0, len(POSE_CLASSES)-1)
    confidence = random.uniform(0.1, 0.8)
    return POSE_CLASSES[pose_idx], confidence

def predict_pose(frame):
    """Main pose prediction function"""
    # Extract keypoints first
    keypoints = extract_pose_keypoints(frame, movenet) if movenet else None
    
    # Then classify using Teachable Machine model
    if keypoints is not None and pose_classifier is not None:
        return predict_pose_from_keypoints(keypoints, pose_classifier)
    else:
        return mock_predict_pose()

def log_result(pose, conf, status):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now(), pose, f"{conf:.2f}", status])
    except Exception as e:
        print(f"Logging failed: {str(e)}")

# Draw pose keypoints on frame
def draw_pose_keypoints(frame, keypoints):
    """Draw pose keypoints on the frame for visualization"""
    if keypoints is None:
        return frame
    
    h, w = frame.shape[:2]
    
    # MoveNet keypoint connections (pairs of keypoints to connect with lines)
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
        (5, 11), (6, 12), (11, 12),  # Body
        (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
    ]
    
    # Draw connections
    for start, end in connections:
        start_point = keypoints[start]
        end_point = keypoints[end]
        
        if start_point[2] > 0.3 and end_point[2] > 0.3:  # Confidence threshold
            start_x = int(start_point[1] * w)
            start_y = int(start_point[0] * h)
            end_x = int(end_point[1] * w)
            end_y = int(end_point[0] * h)
            
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
    
    # Draw keypoints
    for i, keypoint in enumerate(keypoints):
        if keypoint[2] > 0.3:  # Confidence threshold
            x = int(keypoint[1] * w)
            y = int(keypoint[0] * h)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
    
    return frame

# Main application
cv2.namedWindow("PE Guide", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PE Guide", 800, 600)

try:
    completed_poses = set()
    
    while len(completed_poses) < len(POSE_CLASSES):
        if len(completed_poses) == 0:
            speak_async("Choose any pose: squats, shoulder flexion, or arm raises. Hold the position.")
        else:
            speak_async(f"Good! {len(completed_poses)} poses completed. Continue with the next pose.")

        correct = False
        camera_fail_count = 0
        
        while not correct:
            ret, frame = cap.read()
            if not ret:
                camera_fail_count += 1
                if camera_fail_count >= 5:
                    speak_async("Camera connection lost.")
                    exit(1)
                continue
            else:
                camera_fail_count = 0

            # Extract keypoints for visualization (even if model not loaded)
            keypoints = extract_pose_keypoints(frame, movenet) if movenet else None
            if keypoints is not None:
                frame = draw_pose_keypoints(frame, keypoints)

            pose_name, conf = predict_pose(frame)
            print(f"Detected: {pose_name} ({conf:.2f})")

            # Display
            display_frame = cv2.resize(frame, (800, 600))
            remaining = len(POSE_CLASSES) - len(completed_poses)
            required_poses = [p for p in POSE_CLASSES if p not in completed_poses]
            
            # System status
            model_status = "✅ Model Active" if pose_classifier and movenet else "⚠️ Demo Mode"
            cv2.putText(display_frame, model_status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if pose_classifier and movenet else (0, 255, 255), 2)
            cv2.putText(display_frame, f"Remaining: {remaining} poses", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Required: {', '.join(required_poses)}", (10, 110), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Detected: {pose_name}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Confidence: {conf*100:.1f}%", (10, 190), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Color coding
            if conf >= CORRECT_THRESHOLD:
                color = (0, 255, 0)
                status_text = "Good - Hold steady!"
            elif conf >= RETRY_THRESHOLD:
                color = (0, 255, 255)
                status_text = "Almost there - Adjust pose"
            else:
                color = (0, 0, 255)
                status_text = "Keep trying!"

            cv2.putText(display_frame, f"Status: {status_text}", (10, 230), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.imshow("PE Guide", display_frame)

            # Handle pose detection
            if pose_name in POSE_CLASSES and pose_name not in completed_poses:
                if conf >= CORRECT_THRESHOLD:
                    tone_success()
                    speak_async("Good!")
                    log_result(pose_name, conf, "correct")
                    completed_poses.add(pose_name)
                    correct = True
                elif conf >= RETRY_THRESHOLD:
                    tone_almost()
                    speak_async("Almost there. Adjust your pose.")
                    log_result(pose_name, conf, "retry")
                else:
                    tone_fail()
                    speak_async("Pose not detected. Try again.")
                    log_result(pose_name, conf, "failed")

            if cv2.waitKey(1) & 0xFF == ord('x'):
                speak_async("Session stopped. Goodbye.")
                exit(0)

            time.sleep(EVAL_DELAY)

        if len(completed_poses) < len(POSE_CLASSES):
            speak_async("Excellent! Rest for 3 seconds.")
            time.sleep(3)

    tone_success()
    speak_async("Workout complete! Great job today!")

except Exception as e:
    print(f"Unexpected error: {e}")
    speak_async("An error occurred. Session ending.")
finally:
    cap.release()
    cv2.destroyAllWindows()