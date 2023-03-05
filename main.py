import cv2
import numpy as np
from keras.models import model_from_json

emotion_dict = {0: "Angry", 1:"Disgusted",2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

#Loading the json file of the model
json_file = open('data/model_59.json', 'r')
loaded_model = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model)

#Loading weights to the model.
emotion_model.load_weights("data/model_59_weights.h5")
print("Model loaded")

#starting webcam feed
cap = cv2.VideoCapture(0)

# Load the Haar Cascade face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Check if the camera is opened correctly
if not cap.isOpened():
    raise Exception("Could not open video device")

# Loop until the user presses the 'q' key
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame was successfully captured
    if not ret:
        print("Error: failed to capture frame")
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        roi_gray_frame = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)



    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Wait for the user to press a key
    key = cv2.waitKey(1)

    # If the user presses the 'q' key, break the loop
    if key == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()

