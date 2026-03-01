import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "saved_models" / "gesture_model.pkl"
ENCODER_PATH = Path(__file__).resolve().parent / "saved_models" / "encoder.pkl"

class GesturePredictor:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.load()

    def load(self):
        if MODEL_PATH.exists() and ENCODER_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                self.model = pickle.load(f)
            with open(ENCODER_PATH, "rb") as f:
                self.encoder = pickle.load(f)
            print("Model loaded successfully.")
        else:
            print("No trained model found.")

    def predict(self, feature_vector):
        if self.model is None:
            return None, 0.0
        
        X = np.array([feature_vector])
        probs = self.model.predict_proba(X)[0]
        idx = np.argmax(probs)
        conf = float(probs[idx])
        label = self.encoder.inverse_transform([idx])[0]
        
        return label, conf
