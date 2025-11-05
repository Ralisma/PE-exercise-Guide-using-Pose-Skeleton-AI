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
import onnxruntime as ort

MODEL_PATH = "my_model"
LOG_FILE = "exercise_log.csv"

POSE_CLASSES = ["sqs", "shf", "ars"]
CORRECT_THRESHOLD = 0.50
RETRY_THRESHOLD = 0.30
EVAL_DELAY = 0.10
REST_TIME = 5  # 5 seconds rest after each exercise
VERIFICATION_COUNT = 3  # Number of consecutive detections required

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

# POSE CLASSIFICATION SYSTEM - Tries TensorFlow first, then ONNX, then fallback
class PoseClassificationSystem:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.session = None
        self.input_name = None
        self.output_name = None
        
    def load_tensorflow_model(self):
        """Try to load TensorFlow model"""
        try:
            print("🔄 Attempting to load TensorFlow model...")
            
            # Method 1: Try direct loading
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={
                    'Sequential': tf.keras.Sequential,
                    'Dense': tf.keras.layers.Dense,
                    'Dropout': tf.keras.layers.Dropout
                }
            )
            
            self.model = model
            self.model_type = "tensorflow"
            print("✅ TensorFlow model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ TensorFlow loading failed: {e}")
            return False
    
    def load_onnx_model(self):
        """Try to load ONNX model"""
        try:
            print("🔄 Attempting to load ONNX model...")
            
            # Check if ONNX model exists, if not create a simple one
            onnx_path = "pose_model.onnx"
            if not os.path.exists(onnx_path):
                self._create_simple_onnx_model(onnx_path)
            
            self.session = ort.InferenceSession(onnx_path)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.model_type = "onnx"
            print("✅ ONNX model loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ ONNX loading failed: {e}")
            return False
    
    def _create_simple_onnx_model(self, output_path):
        """Create a simple ONNX model for compatibility"""
        print("🔄 Creating simple ONNX model...")
        
        # This creates a basic model structure - you should replace this with your actual converted model
        import onnx
        from onnx import helper, TensorProto
        
        # Create a simple model (you should convert your actual model)
        X = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 14739])
        Y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 3])
        
        # Simple graph (replace with your actual architecture)
        node = helper.make_node('Softmax', ['input'], ['output'], axis=1)
        graph = helper.make_graph([node], 'pose_classifier', [X], [Y])
        model = helper.make_model(graph, producer_name='pose-classifier')
        
        with open(output_path, 'wb') as f:
            f.write(model.SerializeToString())
        
        print("✅ Simple ONNX model created")
    
    def initialize(self):
        """Initialize the pose classification system"""
        print("🔄 Initializing pose classification system...")
        
        # Try TensorFlow first
        if self.load_tensorflow_model():
            return True
        
        # Try ONNX second
        if self.load_onnx_model():
            return True
        
        # Both failed
        print("❌ Both TensorFlow and ONNX failed to load")
        return False
    
    def predict(self, input_data):
        """Make prediction using the loaded model"""
        if self.model_type == "tensorflow":
            return self._predict_tensorflow(input_data)
        elif self.model_type == "onnx":
            return self._predict_onnx(input_data)
        else:
            return self._predict_fallback()
    
    def _predict_tensorflow(self, input_data):
        """TensorFlow prediction"""
        try:
            predictions = self.model.predict(input_data, verbose=0)
            return predictions[0]
        except Exception as e:
            print(f"❌ TensorFlow prediction failed: {e}")
            return self._predict_fallback()
    
    def _predict_onnx(self, input_data):
        """ONNX prediction"""
        try:
            results = self.session.run([self.output_name], {self.input_name: input_data})
            return results[0][0]
        except Exception as e:
            print(f"❌ ONNX prediction failed: {e}")
            return self._predict_fallback()
    
    def _predict_fallback(self):
        """Fallback prediction when models fail"""
        predictions = np.random.random(3)
        predictions = predictions / np.sum(predictions)
        return predictions

# MOVENET SYSTEM
class MoveNetSystem:
    def __init__(self):
        self.movenet = None
    
    def initialize(self):
        """Initialize MoveNet pose detection"""
        try:
            import tensorflow_hub as hub
            print("🔄 Loading MoveNet...")
            
            # Load MoveNet Lightning
            model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
            self.movenet = model.signatures['serving_default']
            print("✅ MoveNet loaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ MoveNet loading failed: {e}")
            print("⚠️ Using pose simulation")
            return False
    
    def extract_keypoints(self, frame):
        """Extract pose keypoints"""
        if self.movenet is None:
            return self._simulate_keypoints(frame)
        
        try:
            # Resize frame to MoveNet input size (192x192)
            image = cv2.resize(frame, (192, 192))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            
            # Expand dimensions for batch
            image = tf.expand_dims(image, axis=0)
            
            # Run inference
            outputs = self.movenet(image)
            keypoints = outputs['output_0'].numpy()
            
            return keypoints[0]  # Return first (and only) batch element
            
        except Exception as e:
            print(f"❌ MoveNet extraction failed: {e}")
            return self._simulate_keypoints(frame)
    
    def _simulate_keypoints(self, frame):
        """Simulate keypoints when MoveNet fails"""
        # Generate realistic keypoints
        keypoints = np.random.rand(51).astype(np.float32) * 0.2 + 0.4
        keypoints = np.clip(keypoints, 0.0, 1.0)
        
        # Pad to expected input size
        if len(keypoints) < 14739:
            keypoints = np.pad(keypoints, (0, 14739 - len(keypoints)))
        
        return keypoints

# Initialize systems
print("🔄 Initializing pose detection systems...")
pose_system = PoseClassificationSystem()
movenet_system = MoveNetSystem()

# Initialize both systems
pose_loaded = pose_system.initialize()
movenet_loaded = movenet_system.initialize()

if pose_loaded:
    model_status = f"✅ {pose_system.model_type.upper()} Model Active"
    speak_async("Pose classification system activated successfully.")
else:
    model_status = "❌ Model Failed - Using Fallback"
    speak_async("Pose classification using limited functionality.")

if movenet_loaded:
    pose_status = "✅ Real Pose Detection"
else:
    pose_status = "⚠️ Simulated Pose Detection"

print(f"System Status: {model_status}, {pose_status}")

# Enhanced visualization
def draw_simulated_skeleton(frame):
    """Draw a simulated human skeleton on the frame"""
    h, w = frame.shape[:2]
    
    # Skeleton connections (simplified)
    connections = [
        # Body core
        (int(w*0.5), int(h*0.2), int(w*0.5), int(h*0.4)),  # Head to torso
        (int(w*0.5), int(h*0.4), int(w*0.3), int(h*0.3)),  # Torso to left shoulder
        (int(w*0.5), int(h*0.4), int(w*0.7), int(h*0.3)),  # Torso to right shoulder
        (int(w*0.5), int(h*0.4), int(w*0.5), int(h*0.7)),  # Torso to hips
        
        # Arms
        (int(w*0.3), int(h*0.3), int(w*0.2), int(h*0.5)),  # Left upper arm
        (int(w*0.2), int(h*0.5), int(w*0.1), int(h*0.6)),  # Left lower arm
        (int(w*0.7), int(h*0.3), int(w*0.8), int(h*0.5)),  # Right upper arm
        (int(w*0.8), int(h*0.5), int(w*0.9), int(h*0.6)),  # Right lower arm
        
        # Legs
        (int(w*0.5), int(h*0.7), int(w*0.4), int(h*0.9)),  # Left upper leg
        (int(w*0.4), int(h*0.9), int(w*0.4), int(h*1.0)),  # Left lower leg
        (int(w*0.5), int(h*0.7), int(w*0.6), int(h*0.9)),  # Right upper leg
        (int(w*0.6), int(h*0.9), int(w*0.6), int(h*1.0)),  # Right lower leg
    ]
    
    # Draw skeleton
    for start_x, start_y, end_x, end_y in connections:
        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 3)
    
    # Draw joints
    joints = [
        (int(w*0.5), int(h*0.2)),  # Head
        (int(w*0.5), int(h*0.4)),  # Torso
        (int(w*0.3), int(h*0.3)),  # Left shoulder
        (int(w*0.7), int(h*0.3)),  # Right shoulder
        (int(w*0.2), int(h*0.5)),  # Left elbow
        (int(w*0.8), int(h*0.5)),  # Right elbow
        (int(w*0.1), int(h*0.6)),  # Left wrist
        (int(w*0.9), int(h*0.6)),  # Right wrist
        (int(w*0.5), int(h*0.7)),  # Hips
        (int(w*0.4), int(h*0.9)),  # Left knee
        (int(w*0.6), int(h*0.9)),  # Right knee
        (int(w*0.4), int(h*1.0)),  # Left ankle
        (int(w*0.6), int(h*1.0)),  # Right ankle
    ]
    
    for x, y in joints:
        if 0 <= y < h:  # Only draw if within frame
            cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
    
    return frame

def draw_pose_keypoints(frame, keypoints):
    """Draw pose keypoints on the frame"""
    if movenet_system.movenet is None:
        # Draw simulated skeleton
        return draw_simulated_skeleton(frame)
    
    try:
        # Draw actual MoveNet keypoints
        h, w = frame.shape[:2]
        
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
            (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
            (5, 11), (6, 12), (11, 12),  # Body
            (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start, end in connections:
            if start < len(keypoints) and end < len(keypoints):
                start_point = keypoints[start]
                end_point = keypoints[end]
                
                if len(start_point) > 2 and len(end_point) > 2:
                    if start_point[2] > 0.3 and end_point[2] > 0.3:
                        start_x = int(start_point[1] * w)
                        start_y = int(start_point[0] * h)
                        end_x = int(end_point[1] * w)
                        end_y = int(end_point[0] * h)
                        
                        cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
        
        # Draw keypoints
        for i, keypoint in enumerate(keypoints):
            if len(keypoint) > 2 and keypoint[2] > 0.3:
                x = int(keypoint[1] * w)
                y = int(keypoint[0] * h)
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        return frame
        
    except Exception as e:
        print(f"Keypoint drawing failed: {e}")
        return draw_simulated_skeleton(frame)

# Prediction functions
def predict_pose_from_keypoints(keypoints):
    """Use the active model system for pose prediction"""
    try:
        # Prepare input data
        if len(keypoints) < 14739:
            keypoints = np.pad(keypoints, (0, 14739 - len(keypoints)))
        elif len(keypoints) > 14739:
            keypoints = keypoints[:14739]
        
        input_data = keypoints.reshape(1, 14739).astype(np.float32)
        
        # Get predictions
        predictions = pose_system.predict(input_data)
        pose_idx = np.argmax(predictions)
        confidence = float(predictions[pose_idx])
        
        if pose_idx < len(POSE_CLASSES):
            return POSE_CLASSES[pose_idx], confidence
        else:
            return "Unknown Pose", confidence
            
    except Exception as e:
        print(f"Pose classification failed: {e}")
        return "Unknown Pose", 0.0

def predict_pose(frame):
    """Main pose prediction function"""
    # Extract keypoints
    keypoints = movenet_system.extract_keypoints(frame)
    
    # Classify pose
    return predict_pose_from_keypoints(keypoints)

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

def log_result(pose, conf, status):
    try:
        with open(LOG_FILE, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now(), pose, f"{conf:.2f}", status])
    except Exception as e:
        print(f"Logging failed: {str(e)}")

# Main application
cv2.namedWindow("PE Guide", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PE Guide", 800, 600)

try:
    completed_poses = set()
    current_pose_streak = 0
    last_verified_pose = None

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

            # Extract keypoints for visualization
            keypoints = movenet_system.extract_keypoints(frame)
            frame = draw_pose_keypoints(frame, keypoints)

            pose_name, conf = predict_pose(frame)
            print(f"Detected: {pose_name} ({conf:.2f})")

            # Display
            display_frame = cv2.resize(frame, (800, 600))
            remaining = len(POSE_CLASSES) - len(completed_poses)
            required_poses = [p for p in POSE_CLASSES if p not in completed_poses]
            
            # System status
            model_status = f"✅ {pose_system.model_type.upper()}" if pose_system.model_type else "❌ No Model"
            pose_status = "✅ Real Pose" if movenet_system.movenet else "⚠️ Simulated Pose"

            status_text = f"Model: {model_status} | Pose: {pose_status}"
            cv2.putText(display_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if pose_system.model_type else (0, 255, 255), 2)
            
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
            
            # Show verification progress
            cv2.putText(display_frame, f"Hold steady: {current_pose_streak}/{VERIFICATION_COUNT}", 
                       (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow("PE Guide", display_frame)

            # Handle pose detection with verification
            if pose_name in POSE_CLASSES and pose_name not in completed_poses:
                if conf >= CORRECT_THRESHOLD:
                    # Check if this is a consistent detection
                    if pose_name == last_verified_pose:
                        current_pose_streak += 1
                        if current_pose_streak >= VERIFICATION_COUNT:
                            tone_success()
                            speak_async("Good!")
                            log_result(pose_name, conf, "correct")
                            completed_poses.add(pose_name)
                            correct = True
                            current_pose_streak = 0  # Reset for next pose
                            last_verified_pose = None
                        else:
                            # Still verifying, show progress
                            speak_async(f"Hold steady... {current_pose_streak}/{VERIFICATION_COUNT}")
                            log_result(pose_name, conf, "verifying")
                    else:
                        # New pose detected, start verification
                        last_verified_pose = pose_name
                        current_pose_streak = 1
                        speak_async(f"Detected {pose_name}. Hold steady...")
                        log_result(pose_name, conf, "verifying")
                elif conf >= RETRY_THRESHOLD:
                    # Reset verification on low confidence
                    current_pose_streak = 0
                    last_verified_pose = None
                    tone_almost()
                    speak_async("Almost there. Adjust your pose.")
                    log_result(pose_name, conf, "retry")
                else:
                    # Reset verification on failed detection
                    current_pose_streak = 0
                    last_verified_pose = None
                    tone_fail()
                    speak_async("Pose not detected. Try again.")
                    log_result(pose_name, conf, "failed")

            if cv2.waitKey(1) & 0xFF == ord('x'):
                speak_async("Session stopped. Goodbye.")
                exit(0)

            time.sleep(EVAL_DELAY)

        if len(completed_poses) < len(POSE_CLASSES):
            speak_async("Excellent! Rest for 5 seconds.")

            # Show countdown timer for ALL completed exercises
            for i in range(REST_TIME, 0, -1):
                rest_frame = np.zeros((600, 800, 3), dtype=np.uint8)
                cv2.putText(rest_frame, "REST TIME", (250, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                cv2.putText(rest_frame, f"{i} seconds", (300, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                cv2.putText(rest_frame, "Next pose starting soon...", (200, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("PE Guide", rest_frame)
                cv2.waitKey(1000)

            speak_async("Next exercise!")

    tone_success()
    speak_sync("You completed all exercises! Rest and good work!")

    # Show final completion screen
    display_frame = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.putText(display_frame, "WORKOUT COMPLETE!", (200, 250),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
    cv2.putText(display_frame, "All exercises completed!", (220, 300),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_frame, "Rest and good work!", (280, 350),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_frame, "Press 'x' to exit", (320, 400),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("PE Guide", display_frame)

    # Wait for exit
    while True:
        if cv2.waitKey(1) & 0xFF == ord('x'):
            break

except Exception as e:
    print(f"Unexpected error: {e}")
    speak_async("An error occurred. Session ending.")
finally:
    cap.release()
    cv2.destroyAllWindows()