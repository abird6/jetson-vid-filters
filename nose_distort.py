import numpy as np
import cv2

# initialise CSI port for camera
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def enlarge_nose(frame, faces):
    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Enlarge the nose region
        nose_x = x + w // 4 + 25
        nose_y = y + h // 2 - 5
        nose_width = w // 2 - 50
        nose_height = h // 3 - 20

        # Ensure the nose region is within the frame boundaries
        nose_x = max(0, nose_x)
        nose_y = max(0, nose_y)
        nose_width = min(frame.shape[1] - nose_x, nose_width)
        nose_height = min(frame.shape[0] - nose_y, nose_height)

        # Resize the nose region
        nose_region = frame[nose_y:nose_y + nose_height, nose_x:nose_x + nose_width]

        # Enlarge the nose region
        enlarged_nose = cv2.resize(nose_region, (nose_width * 2, nose_height * 2), interpolation=cv2.INTER_LINEAR)

        # Replace the original nose region with the enlarged one
        frame[y + 50:y + enlarged_nose.shape[0] + 50, x + 50:x + enlarged_nose.shape[1] + 50] = enlarged_nose

    return frame

if __name__ == '__main__':
    # set up face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # read in video feed
    vid = cv2.VideoCapture(gstreamer_pipeline(flip_method=q), cv2.CAP_GSTREAMER)

    ret, frame = vid.read()

    while True:
        ret, frame = vid.read()

        # find faces in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )


        for (x, y, w, h) in faces:
            # Apply nose enlargement effect
            frame = enlarge_nose(frame, faces)

        # Display the result
        cv2.imshow('Nose Enlargement', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
