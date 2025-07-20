import cv2
import numpy as np
import tensorflow as tf

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
ensemble_model = tf.keras.models.load_model('best_model.keras')
class_names = ['Drowsy', 'Non drowsy'] # Hardcoded instead of json.load

# Start webcam capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, 'Face', (x, y - 10), font, 1, (0, 255, 0), 2)

    # Preprocess frame for model prediction
    img = cv2.resize(frame, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0 # model was trained on images in [0-1] scale

    raw_score = ensemble_model.predict(img)
    bin_score = (raw_score >= 0.5).astype(int).item()
    predicted_class = class_names[bin_score]
    raw_score_disp = np.round(raw_score.item(), 4) # For easier display

    # Terminal output
    color = "\033[1;32m" if raw_score >= 0.5 else "\033[1;31m" # Red color if drowsy and green if not
    print(f"{color}Raw score: {raw_score_disp}, Predicted: {predicted_class}\033[0m")

    # GUI overlay texts
    text1 = "Raw score: " + str(raw_score_disp)
    text2 = "Predicted: " +  predicted_class
    color = (0, 0, 255) if predicted_class == 'Drowsy' else (0, 255, 0) # color order is BGR

    cv2.putText(frame, text1, (50, 50), font, 1, color, 2, cv2.LINE_AA)
    cv2.putText(frame, text2, (50, 100), font, 1, color, 2, cv2.LINE_AA)

    # Display the frame
    win_title = 'Real-time Driver Drowsiness Detection'
    cv2.namedWindow(win_title, cv2.WINDOW_NORMAL)
    cv2.imshow(win_title, frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break

cap.release()
cv2.destroyAllWindows()