import os, cv2, numpy as np
import simpleaudio as sa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress messy terminal messages; must be declared before importing tf
import tensorflow as tf

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

font = cv2.FONT_HERSHEY_SIMPLEX
ensemble_model = tf.keras.models.load_model('example.keras')
class_names = ['Drowsy', 'Non drowsy']

# Alert sound effect for drowsiness
wave_obj = sa.WaveObject.from_wave_file("alert.wav")
play_obj = None

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

    flipped_frame = cv2.flip(frame, 1)
    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(flipped_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("\033[1;33mNo face detected\033[0m")

    # Draw bounding boxes around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(flipped_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(flipped_frame, 'subject', (x, y - 10), font, 1, (0, 255, 0), 2)

        roi_color = flipped_frame[y:y+ h, x:x + w]    # Cropping region of interest i.e. face area from frame
        roi_color = cv2.resize(roi_color, (224, 224))
        img = tf.keras.preprocessing.image.img_to_array(roi_color)
        img = tf.cast(img, tf.float32) / 255.0   # Normalizing to match training standard
        img = np.expand_dims(img, axis=0)

        raw_score = ensemble_model.predict(img)
        bin_score = int(raw_score >= 0.5)
        predicted_class = class_names[bin_score]
        raw_score_disp = round(float(raw_score), 4)  # Round score for display

        # Play sound if the predicted class is 'Drowsy'
        if predicted_class == "Drowsy":
            if not play_obj or not play_obj.is_playing():
                play_obj = wave_obj.play()
        else:
            play_obj = None

        # Terminal output
        color = "\033[1;32m" if raw_score >= 0.5 else "\033[1;31m"  # Red color if drowsy and green if not
        print(f"{color}Raw score: {raw_score_disp}, Predicted: {predicted_class}\033[0m")

        # GUI overlay
        color = (0, 0, 255) if predicted_class == 'Drowsy' else (0, 255, 0)  # color order is BGR
        cv2.putText(flipped_frame, f"Raw score: {raw_score_disp}", (10, 20), font, 0.5, color, 1, cv2.LINE_AA)
        cv2.putText(flipped_frame, f"Predicted: {predicted_class}", (10, 40), font, 0.5, color, 1, cv2.LINE_AA)

    # Display the frame
    cv2.namedWindow('Real-time Driver Drowsiness Detection', cv2.WINDOW_NORMAL)
    cv2.imshow('Real-time Driver Drowsiness Detection', flipped_frame)

    if cv2.waitKey(1) & 0XFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
