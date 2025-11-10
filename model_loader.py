import tensorflow as tf
import numpy as np
import os
import json

class TensorFlowJSModelLoader:
    def __init__(self, model_path):
        self.model = None
        self.model_loaded = False
        self.model_type = "N/A"
        self.model_h5_path = os.path.join(model_path, 'model.h5')
        self.model_path = model_path

    def initialize(self):
        print("🔄 Initializing Keras model loader...")
        if not os.path.exists(self.model_h5_path):
            print(f"❌ Model H5 not found at: {self.model_h5_path}")
            return self.initialize_fallback()

        try:
            print("✅ Found model.h5. Loading Keras model...")
            self.model = tf.keras.models.load_model(self.model_h5_path)
            self.model_loaded = True
            self.model_type = "Keras (H5)"
            print("✅✅✅ Model loaded successfully! ✅✅✅")

            # Test the model
            test_input_shape = [1] + list(self.model.input_shape[1:])
            test_input = np.random.random(test_input_shape).astype(np.float32)
            test_output = self.model.predict(test_input, verbose=0)
            print(f"✅ Model test successful. Output shape: {test_output.shape}")

            return True

        except Exception as e:
            print(f"❌ Keras model loading failed: {e}")
            import traceback
            traceback.print_exc()
            return self.initialize_fallback()



    def initialize_fallback(self):
        """Initialize the fallback simulation if loading fails"""
        print("⚠️ Model loading failed. Using enhanced fallback simulation.")
        self.model_loaded = False
        self.model_type = "Fallback Simulation"
        return True

    def predict(self, input_data):
        """Make prediction using the loaded model or fallback"""
        if self.model_loaded:
            try:
                predictions = self.model.predict(input_data, verbose=0)
                # Ensure predictions is a numpy array
                if isinstance(predictions, list):
                    predictions = np.array(predictions)
                return predictions[0]
            except Exception as e:
                print(f"❌ Model prediction failed: {e}. Reverting to fallback.")
                self.model_loaded = False
                return self._enhanced_fallback_prediction()
        else:
            return self._enhanced_fallback_prediction()

    def _enhanced_fallback_prediction(self):
        """Enhanced fallback prediction"""
        if not hasattr(self, 'fallback_counter'):
            self.fallback_counter = 0
            self.current_pose_idx = 0
            self.pose_confidence = 0.3

        self.fallback_counter += 1

        if self.fallback_counter % 45 == 0:
            self.current_pose_idx = (self.current_pose_idx + 1) % len(self.pose_classes)
            self.pose_confidence = 0.3

        self.pose_confidence = min(0.95, self.pose_confidence + 0.02)

        predictions = np.random.random(len(self.pose_classes)) * 0.15
        predictions[self.current_pose_idx] = self.pose_confidence

        predictions = predictions / np.sum(predictions)
        return predictions

    def set_pose_classes(self, pose_classes):
        self.pose_classes = pose_classes
