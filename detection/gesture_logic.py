import numpy as np

def is_distress_signal(landmarks):
    """
    Heuristic for the 'Signal for Help' (Distress Signal).
    Detects the final state: Fist with thumb tucked under fingers.
    """
    if landmarks is None or len(landmarks) < 21:
        return False, 0.0

    # 1. MCP joints for reference (palm)
    mcp_indices = [5, 9, 13, 17]
    palm_center_x = np.mean([landmarks[i][0] for i in mcp_indices])
    
    # 2. Check if fingers are closed (Tips 8, 12, 16, 20 are below PIP joints 6, 10, 14, 18)
    # MediaPipe Y increases downwards.
    fingers_closed = True
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        if landmarks[tip][1] < landmarks[pip][1]: # Tip is above PIP (Extended)
            fingers_closed = False
            break
    
    # 3. Check if thumb (4) is tucked
    # Simple check: Thumb tip is below index MCP
    thumb_tucked = landmarks[4][1] > landmarks[5][1]
    
    if fingers_closed and thumb_tucked:
        return True, 0.9
        
    return False, 0.0
