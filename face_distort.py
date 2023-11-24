import numpy as np
import cv2
import math

# initialise CSI port for camera
def gstreamer_pipeline(
    sensor_id=0,
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=1,
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

if __name__ == '__main__':
    # Parameters for distortion
    center_x_init = 320  # Adjust based on your camera's resolution
    center_y_init = 240  # Adjust based on your camera's resolution
    radius = 100
    scale_x = 1.0
    scale_y = 1.0
    amount = 0.5

    # set up face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # read in video feed
    vid = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

    ret, frame = vid.read()

    # grab the dimensions of the image
    (h, w, _) = frame.shape

    while True:
        ret, frame = vid.read()

        # find faces in frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags = cv2.CASCADE_SCALE_IMAGE #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        if len(faces) > 0:

            # faces: x y h w
            face_center_x = faces[0][0] + faces[0][3] // 2
            face_center_y = faces[0][1] + faces[0][2] // 2

            # set up the x and y maps as float32
            flex_x = np.zeros((h, w), np.float32)
            flex_y = np.zeros((h, w), np.float32)

            # create map with the barrel pincushion distortion formula
            for y in range(h):
                delta_y = scale_y * (y - face_center_y)
                for x in range(w):
                    # determine if pixel is within an ellipse
                    delta_x = scale_x * (x - face_center_x)
                    distance = delta_x * delta_x + delta_y * delta_y
                    if distance >= (radius * radius):
                        flex_x[y, x] = x
                        flex_y[y, x] = y
                    else:
                        factor = 1.0
                        if distance > 0.0:
                            factor = math.pow(math.sin(math.pi * math.sqrt(distance) / radius / 2), -amount)
                        flex_x[y, x] = factor * delta_x / scale_x + face_center_x
                        flex_y[y, x] = factor * delta_y / scale_y + face_center_y

            # do the remap  this is where the magic happens
            dst = cv2.remap(frame, flex_x, flex_y, cv2.INTER_LINEAR)

            cv2.imshow('Distorted', dst)

        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()



