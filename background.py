# this file contains resources and functions necessary for background blurring and replacement
#
# Author:           Anthony Bird    Szymon Borkowski
# Organisation:     University of Galway
# Project Name:     EE551 - DIY Video Filters for Jetson Nano
# Date:             3 Nov 2023

# import libraries
import cv2
import numpy as np

# detects background of frame
def detect_bg():
    return 0

########################
# Testing & Debugging
########################

if __name__ == '__main__':
    # read in video feed
    vid = cv2.VideoCapture(0)

    background_model = None

    frame_count = 0

    sensitivity = 60

    while True:
        ret, frame = vid.read()

        if not ret:
            break

        # convert frame to greyscale
        frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # initialise background model
        if background_model is None:
            background_model = frame_grey.copy().astype(np.float32)
            continue

        # find absolute difference between current frame and background
        frame_delta = cv2.absdiff(background_model.astype(np.uint8), frame_grey)

        # create binary mask
        thresh = cv2.threshold(frame_delta, sensitivity, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # find biggest contour
        if contours:
            print("Drawing contours")

            contour_large = max(contours, key=cv2.contourArea)

            # Draw an outline around the largest contour
            cv2.drawContours(frame, [contour_large], -1, (0, 255, 0), 2)

            # create mask for foreground
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour_large], -1, 255, -1)

            # invert mask to get background
            mask = cv2.bitwise_not(mask)

            # apply blurring to background
            bg_blur = cv2.GaussianBlur(frame, (0, 0), 5)
            frame[mask == 0] = bg_blur[mask == 0]

            # slow down updates of background model
            if frame_count < sensitivity:
                frame_count += 1
            else:
                background_model = frame_grey.copy().astype(np.float32)
                frame_count = 0


        cv2.imshow('Background Blur', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vid.release()
    cv2.destroyAllWindows()
