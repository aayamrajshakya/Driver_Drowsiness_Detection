import os, dlib, cv2, shutil, numpy as np
from imutils import face_utils

# EAR thresholds
OPEN_EAR_THRESHOLD = 0.40  # above this: considered open eyes
CLOSED_EAR_THRESHOLD = 0.18  # below this: considered closed eyes

# Paths
paths = {
    "drowsy": {
        "input": "dataset/drowsy",
        "output": "dataset/open_eye_in_drowsy",
        "condition": lambda ear: ear > OPEN_EAR_THRESHOLD,
        "label": "OPEN"
    },
    "non_drowsy": {
        "input": "dataset/non-drowsy",
        "output": "dataset/closed_eye_in_nondrowsy",
        "condition": lambda ear: ear < CLOSED_EAR_THRESHOLD,
        "label": "CLOSED"
    }
}

# Load face detector and shape predictor
predictor_path = 'shape_predictor_68_face_landmarks.dat'
if not os.path.exists(predictor_path):
    raise FileNotFoundError(f"Missing {predictor_path}")

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# EAR calculation function
def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# Landmark indices
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Process both sets
for key, info in paths.items():
    os.makedirs(info["output"], exist_ok=True)
    print(f"\nProcessing {key} set...")

    for filename in os.listdir(info["input"]):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(info["input"], filename)
            image = cv2.imread(img_path)
            if image is None:
                print(f"[SKIPPED] Could not read {filename}")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            moved = False
            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if info["condition"](ear):
                    print(f"[MOVED] {filename} likely has {info['label']} eyes (EAR={ear:.2f})")
                    shutil.move(img_path, os.path.join(info["output"], filename))
                    moved = True
                    break  # Process only one face per image

            if not moved:
                print(f"[OK] {filename} didn't match criteria")

print("\nAll processing complete.")
