import numpy as np

def vector(a, b):
    """Vector from a -> b"""
    return np.array([b.x - a.x, b.y - a.y, b.z - a.z])

def normalize(v):
    """Normalize a vector, avoid division by zero"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def angle_between(v1, v2):
    """Angle in radians between two vectors"""
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(dot)

def compute_hand_features(hand_3d, prev_tips=None):
    """
    Converts MediaPipe 3D hand landmarks to feature vector.
    hand_3d: list of 21 world landmarks
    prev_tips: previous frame fingertip positions (for velocity)
    Returns:
        feature_vector (list)
        current_tips (numpy array)
    """
    feature_vector = []

    fingers = {
        "thumb": [1,2,3,4],
        "index": [5,6,7,8],
        "middle": [9,10,11,12],
        "ring": [13,14,15,16],
        "pinky": [17,18,19,20]
    }

    wrist = hand_3d[0]

    # Bone vectors + angles
    for idx in fingers.values():
        mcp = hand_3d[idx[0]]
        pip = hand_3d[idx[1]]
        dip = hand_3d[idx[2]]
        tip = hand_3d[idx[3]]

        v1 = vector(wrist, mcp)
        v2 = vector(mcp, pip)
        v3 = vector(pip, dip)
        v4 = vector(dip, tip)

        feature_vector.extend(normalize(v1))
        feature_vector.extend(normalize(v2))
        feature_vector.extend(normalize(v3))
        feature_vector.extend(normalize(v4))

        feature_vector.append(angle_between(v2, v3))
        feature_vector.append(angle_between(v3, v4))

    # Fingertip velocity
    current_tips = np.array([[hand_3d[i].x, hand_3d[i].y, hand_3d[i].z] for i in [4,8,12,16,20]])
    if prev_tips is not None:
        velocity = current_tips - prev_tips
        feature_vector.extend(velocity.flatten())
    else:
        feature_vector.extend(np.zeros(15))

    return feature_vector, current_tips