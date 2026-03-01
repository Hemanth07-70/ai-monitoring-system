import numpy as np

def landmarks_to_features(landmarks):
    """
    landmarks: (21, 3)
    returns: (63,) normalized feature vector
    """
    # 1. Flatten
    coords = landmarks.flatten() # 63
    
    # 2. Normalize by wrist (Landmark 0)
    # MediaPipe landmarks are already 0-1 relative to image, 
    # but we want them relative to the hand.
    wrist = landmarks[0]
    normalized = landmarks - wrist
    
    # 3. Scale by maximum distance to normalize hand size
    max_dist = np.max(np.linalg.norm(normalized, axis=1))
    if max_dist > 0:
        normalized = normalized / max_dist
        
    return normalized.flatten()
