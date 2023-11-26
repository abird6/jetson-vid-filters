import cv2
import sys
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=2), cv2.CAP_GSTREAMER)

# Method for performing Laplacian edge detection
def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges

# Determine criteria
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

# Implementing K-Means
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

while True:
    #capture frame by frame
    ret,frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE #flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    #Draw a rectangle around the faces
    if len(faces) > 0:
      for (x, y, w,h) in faces:
          # Extract face
          face = frame[y:y+h, x:x+w]
          
          # Run Laplacian edge detection
          edges = edge_mask(face, 11, 7)

          # Reduce the colour space to only 3 colours
          colour = np.float32(face).reshape((-1, 3))

          # Blur the colours to smooth out the edges
          blurred = cv2.bilateralFilter(colour, d=7, sigmaColor=200, sigmaSpace=200)
          
          # AND together the edges and the colours
          cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)

          # Overlay result into original image
          frame[y:y+h, x:x+w] = cartoon
      
    cv2.imshow('Cartoon Filter',frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()