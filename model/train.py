import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import sys

# Get project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from data.preprocess import landmarks_to_features

def train_model():
    dataset_dir = ROOT / "dataset"
    if not dataset_dir.exists():
        print("Dataset directory not found. Please collect data first.")
        return

    print("Loading dataset...")
    X = []
    y = []

    for gesture_folder in dataset_dir.iterdir():
        if gesture_folder.is_dir():
            label = gesture_folder.name
            print(f"  Processing '{label}'...")
            for npy_file in gesture_folder.glob("*.npy"):
                landmarks = np.load(npy_file)
                features = landmarks_to_features(landmarks)
                X.append(features)
                y.append(label)

    if not X:
        print("No data found in dataset/ folder.")
        return

    X = np.array(X)
    y = np.array(y)

    # Encode labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    print(f"Training on {len(X_train)} samples...")
    # Using SVC as it's robust for landmark classification
    model = SVC(probability=True, kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nTraining Results:")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

    # Save Model
    model_dir = ROOT / "model" / "saved_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    with open(model_dir / "gesture_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(model_dir / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)

    print(f"\nModel saved to {model_dir}/gesture_model.pkl")

if __name__ == "__main__":
    train_model()
