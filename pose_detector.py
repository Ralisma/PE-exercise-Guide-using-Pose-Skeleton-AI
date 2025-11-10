import cv2
import numpy as np
import mediapipe as mp

class PoseDetectionSystem:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.last_keypoints = np.zeros(51)  # 17 keypoints * 3 values
        self.last_results = None

        self.pose_indices = [
            0,  # Nose
            2,  # Left Eye
            5,  # Right Eye
            7,  # Left Ear
            8,  # Right Ear
            11, # Left Shoulder
            12, # Right Shoulder
            13, # Left Elbow
            14, # Right Elbow
            15, # Left Wrist
            16, # Right Wrist
            23, # Left Hip
            24, # Right Hip
            25, # Left Knee
            26, # Right Knee
            27, # Left Ankle
            28  # Right Ankle
        ]

    def initialize(self):
        print("🔄 Initializing MediaPipe pose detection...")
        print("✅ MediaPipe ready")
        return True

    def extract_keypoints(self, frame):
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            self.last_results = self.pose.process(image)

            if self.last_results.pose_landmarks:
                keypoints = []
                landmarks = self.last_results.pose_landmarks.landmark

                for i in self.pose_indices:
                    lm = landmarks[i]
                    keypoints.extend([lm.y, lm.x, lm.visibility])

                self.last_keypoints = np.array(keypoints)

        except Exception as e:
            print(f"MediaPipe processing error: {e}")

        keypoints_51 = self.last_keypoints

        # Flatten to 1D array of 14739 elements
        keypoints_flattened = keypoints_51.flatten()

        if len(keypoints_flattened) < 14739:
            keypoints_padded = np.pad(keypoints_flattened, (0, 14739 - len(keypoints_flattened)))
        else:
            keypoints_padded = keypoints_flattened[:14739]

        return keypoints_padded

    def draw_landmarks(self, frame):
        if self.last_results and self.last_results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                self.last_results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            )
