import cv2
import sys
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

replacement_image = cv2.imread('cat2.png', cv2.IMREAD_UNCHANGED)

while True:
    # Capture frame by frame
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        replacement_face = cv2.resize(replacement_image, (w, h))
        alpha_channel = replacement_face[:, :, 3] / 255.0  # Normalize alpha channel
        beta = 1.0 - alpha_channel

        # Overlay the replacement face on the original frame
        for c in range(0, 3):
            frame[y:y+h, x:x+w, c] = (beta * frame[y:y+h, x:x+w, c] +
                                       alpha_channel * replacement_face[:, :, c])

    cv2.imshow('video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
