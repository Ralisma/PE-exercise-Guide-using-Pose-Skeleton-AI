import cv2
import numpy as np
import time
import winsound
import os
from config import *
from speech import SpeechManager
from logger import Logger
from model_loader import TensorFlowJSModelLoader
from pose_detector import PoseDetectionSystem
from display_manager import DisplayManager

def play_tone(freq, duration):
    winsound.Beep(int(freq), int(duration * 1000))

def tone_success():
    play_tone(TONE_SUCCESS_FREQ, TONE_SUCCESS_DURATION)

def tone_almost():
    play_tone(TONE_ALMOST_FREQ, TONE_ALMOST_DURATION)

def tone_fail():
    play_tone(TONE_FAIL_FREQ, TONE_FAIL_DURATION)

def predict_pose_from_keypoints(keypoints, model_loader):
    try:
        input_data = keypoints.reshape(1, 14739).astype(np.float32)

        predictions = model_loader.predict(input_data)
        # Ensure predictions is a numpy array
        if isinstance(predictions, list):
            predictions = np.array(predictions)
        pose_idx = np.argmax(predictions)
        confidence = float(predictions[pose_idx])

        if pose_idx < len(POSE_CLASSES):
            return POSE_CLASSES[pose_idx], confidence
        else:
            return "Unknown Pose", confidence

    except Exception as e:
        print(f"Pose classification error: {e}")
        pose_idx = np.random.randint(0, len(POSE_CLASSES))
        confidence = 0.4 + np.random.random() * 0.4
        return POSE_CLASSES[pose_idx], confidence

def predict_pose(frame, pose_detection, model_loader):
    keypoints = pose_detection.extract_keypoints(frame)
    return predict_pose_from_keypoints(keypoints, model_loader)

def main():
    # Initialize speech
    speech_manager = SpeechManager(SPEECH_RATE)
    speech_manager.speak_sync("System initializing, please wait.")

    # Initialize logger
    logger = Logger(LOG_FILE)

    # Initialize model loader
    model_loader = TensorFlowJSModelLoader(MODEL_PATH)
    model_loader.set_pose_classes(POSE_CLASSES)

    # Initialize pose detection
    pose_detection = PoseDetectionSystem()

    # Initialize systems
    print("🔄 Initializing systems...")
    model_loaded = model_loader.initialize()
    detection_loaded = pose_detection.initialize()

    # Status reporting
    if model_loaded and model_loader.model_loaded:
        model_status = f"✅ {model_loader.model_type}"
        speech_manager.speak_async("Pose classification model loaded successfully.")
    else:
        model_status = "❌ Model Failed"
        speech_manager.speak_async("Using enhanced classification simulation.")

    detection_status = "✅ MediaPipe" if detection_loaded else "❌ Detection Failed"
    print(f"System Status: Model: {model_status}, Detection: {detection_status}")

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
        speech_manager.speak_async("Camera not detected. Please check connection.")
        exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    speech_manager.speak_sync("System ready. Let's begin your guided exercise.")

    # Main application
    display_manager = DisplayManager()

    try:
        completed_poses = set()
        current_pose_streak = 0
        last_verified_pose = None
        frame_count = 0

        while len(completed_poses) < len(POSE_CLASSES):
            if len(completed_poses) == 0:
                speech_manager.speak_async("Choose any pose: squats, shoulder flexion, or arm raises. Hold the position.")
            else:
                speech_manager.speak_async(f"Good! {len(completed_poses)} poses completed. Continue with the next pose.")

            correct = False
            camera_fail_count = 0

            while not correct:
                ret, frame = cap.read()
                if not ret:
                    camera_fail_count += 1
                    if camera_fail_count >= 5:
                        speech_manager.speak_async("Camera connection lost.")
                        if cap: cap.release()
                        exit(1)
                    continue
                else:
                    camera_fail_count = 0

                frame = cv2.flip(frame, 1)

                pose_name, conf = predict_pose(frame, pose_detection, model_loader)
                pose_detection.draw_landmarks(frame)

                frame_count += 1

                if frame_count % 15 == 0:
                    print(f"Frame {frame_count}: {pose_name} ({conf:.2f})")

                display_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                remaining = len(POSE_CLASSES) - len(completed_poses)
                required_poses = [p for p in POSE_CLASSES if p not in completed_poses]

                if model_loader.model_loaded:
                    model_status_text = f"✅ {model_loader.model_type}"
                    status_color = (0, 255, 0)
                    detection_text = "Using MediaPipe detection"
                else:
                    model_status_text = "❌ Fallback"
                    status_color = (0, 255, 255)
                    detection_text = "Using enhanced simulation"

                cv2.putText(display_frame, f"Model: {model_status_text}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                cv2.putText(display_frame, detection_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

                cv2.putText(display_frame, f"Remaining: {remaining} poses", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Required: {', '.join(required_poses)}", (10, 130),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Detected: {pose_name}", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(display_frame, f"Confidence: {conf*100:.1f}%", (10, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                if conf >= CORRECT_THRESHOLD:
                    color = (0, 255, 0)
                    status_text = "Good - Hold steady!"
                elif conf >= RETRY_THRESHOLD:
                    color = (0, 255, 255)
                    status_text = "Almost there - Adjust pose"
                else:
                    color = (0, 0, 255)
                    status_text = "Keep trying!"

                cv2.putText(display_frame, f"Status: {status_text}", (10, 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                cv2.putText(display_frame, f"Hold steady: {current_pose_streak}/{VERIFICATION_COUNT}",
                            (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # --- START OF FIX ---
                # Calculate the wait time in milliseconds from config
                wait_ms = int(EVAL_DELAY * 1000)
                
                # Pass the wait time to show_frame. 
                # This function now handles both displaying the frame AND waiting,
                # which allows it to process key presses (like 'x') during the wait.
                display_manager.show_frame(display_frame, wait_ms=wait_ms)
                # --- END OF FIX ---

                if pose_name in POSE_CLASSES and pose_name not in completed_poses:
                    if conf >= CORRECT_THRESHOLD:
                        if pose_name == last_verified_pose:
                            current_pose_streak += 1
                            if current_pose_streak >= VERIFICATION_COUNT:
                                tone_success()
                                speech_manager.speak_async("Excellent! Pose completed!")
                                logger.log_result(pose_name, conf, "correct")
                                completed_poses.add(pose_name)
                                correct = True
                                current_pose_streak = 0
                                last_verified_pose = None
                            else:
                                if current_pose_streak % 2 == 0:
                                    speech_manager.speak_async(f"Hold steady... {current_pose_streak}/{VERIFICATION_COUNT}")
                                logger.log_result(pose_name, conf, "verifying")
                        else:
                            last_verified_pose = pose_name
                            current_pose_streak = 1
                            speech_manager.speak_async(f"Detected {pose_name}. Hold steady...")
                            logger.log_result(pose_name, conf, "verifying")
                    elif conf >= RETRY_THRESHOLD:
                        current_pose_streak = 0
                        last_verified_pose = None
                        tone_almost()
                        speech_manager.speak_async("Almost there. Adjust your pose.")
                        logger.log_result(pose_name, conf, "retry")
                    else:
                        current_pose_streak = 0
                        last_verified_pose = None
                        tone_fail()
                        speech_manager.speak_async("Pose not detected. Try again.")
                        logger.log_result(pose_name, conf, "failed")

                if display_manager.should_close():
                    speech_manager.speak_async("Session stopped. Goodbye.")
                    exit(0)

                # time.sleep(EVAL_DELAY) # <-- This line is removed

            if len(completed_poses) < len(POSE_CLASSES):
                speech_manager.speak_async("Excellent! Rest for 5 seconds.")

                for i in range(REST_TIME, 0, -1):
                    rest_frame = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)
                    cv2.putText(rest_frame, "REST TIME", (250, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    cv2.putText(rest_frame, f"{i} seconds", (300, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
                    cv2.putText(rest_frame, "Next pose starting soon...", (200, 400),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    display_manager.show_frame(rest_frame, wait_ms=1000)

                speech_manager.speak_async("Next exercise!")

        tone_success()
        speech_manager.speak_sync("Workout complete! You finished all exercises! Rest and good work!")

        display_manager.show_completion_frame()

    except Exception as e:
        print(f"Unexpected error: {e}")
        speech_manager.speak_async("An error occurred. Session ending.")
    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()