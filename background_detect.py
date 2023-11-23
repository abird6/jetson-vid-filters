# import libraries
import numpy as np
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import pandas as pd
import seaborn as sns
from skimage.io import imread
from skimage.color import rgb2ycbcr, gray2rgb

# initialise skin cascade classifier and GMM
def init_skin_classifier(path):
    print('Initialising skin/not skin classifier...')

    # load classifier text file
    df = pd.read_csv(path, header=None, delim_whitespace=True)
    df.columns = ['B', 'G', 'R', 'skin']

    # convert to YCbCr colour space
    df['Cb'] = np.round(128 -.168736*df.R -.331364*df.G + .5*df.B).astype(int)
    df['Cr'] = np.round(128 +.5*df.R - .418688*df.G - .081312*df.B).astype(int)
    df.drop(['B','G','R'], axis=1, inplace=True)

    # setup GMM (Gaussian Mixture Model)
    print('Creating GMM...')
    skin_data = df[df.skin==1].drop(['skin'], axis=1).to_numpy()
    not_skin_data = df[df.skin==2].drop(['skin'], axis=1).to_numpy()

    # create GMM for each class
    skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(skin_data)
    not_skin_gmm = GaussianMixture(n_components=4, covariance_type='full').fit(not_skin_data)
    print('GMM created successfully')

    return skin_gmm, not_skin_gmm

def get_skin_mask(frame_rgb, skin_roi, skin_gmm, not_skin_gmm):
    # create rough mask around skin_roi (i.e. face)
    mask_roi = np.zeros_like(frame_rgb[:, :, 0])
    mask_roi[skin_roi[2]:skin_roi[3], skin_roi[0]:skin_roi[1]] = 255
    frame = cv2.bitwise_and(frame_rgb, frame_rgb, mask=mask_roi)

    # convert to YCbCr
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    frame = np.reshape(frame, (-1, 3))

    # generate mask
    skin_score = skin_gmm.score_samples(frame[..., 1:])
    not_skin_score = not_skin_gmm.score_samples(frame[..., 1:])

    # optimise mask
    mask = skin_score > not_skin_score
    mask = mask.reshape(frame_rgb.shape[0], frame_rgb.shape[1])
    mask = mask.astype(np.uint8)
    mask = cv2.dilate(mask, (9, 9), iterations=20)

    return mask

def get_shoulder_mask(frame, x, y, w, h):
    shoulder_mask = []

    k = 2.5 # scaling factor from head to shoulders
    center_x = x + (w/2)
    w2 = int(k * w)
    x2 = int(center_x - ((w2)/2))
    y2 = int(y + h)
    h2 = int(frame.shape[1] - y2)

    # create mask for shoulders region
    region_mask = np.zeros_like(frame[:, :, 0])
    region_mask[y2:y2+h2, x2:x2+w2] = 255

    # apply colour segmentation to mask region
    frame_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    mean_colour = np.mean(frame_yuv[y2:y2+h2, x2:x2+w2], axis=(0, 1)).astype(np.uint8)
    tolerance = 50
    lower = np.array([mean_colour[0] - tolerance, mean_colour[1] - tolerance, mean_colour[2] - tolerance])
    upper = np.array([mean_colour[0] + tolerance, mean_colour[1] + tolerance, mean_colour[2] + tolerance])


    # create new mask from hsv colour limits
    hsv_mask = cv2.inRange(frame_yuv, lower, upper)
    shoulder_mask = cv2.bitwise_and(hsv_mask, region_mask)

    # dilate mask to reduce blank spots
    shoulder_mask = cv2.dilate(shoulder_mask, (10, 10), iterations=40)
        
    return shoulder_mask

def get_inner_face_mask(frame, eyes):
    # extract mask from left-side eye
    left_eye = eyes[0] if len(eyes) < 2 or eyes[1][0] > eyes[0][0] else eyes[1]

    # extract bounding box coordinates from left eye
    x, y, w, h = left_eye[0], left_eye[1], left_eye[2], left_eye[3] 

    # calculate new coordinates with scaling factor
    scale = 2.5
    w1 = int(scale * w)
    h1 = int(scale * h)
    x1 = x + w1
    y1 = y + h1

    # use box coordinates to create mask
    face_mask = np.zeros_like(frame[:, :, 0])
    face_mask[y:y1, x:x1] = 255
    print('[1/3] Inner face mask done')

    return face_mask



# setup cascade classifier for face region
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
print('Initialising face classifier...')

# setup classifier for eye region
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
print('Initialising eye classifier...')

# setup skin_notSkin classifier
path = 'Skin_NonSkin.txt'
skin_gmm, not_skin_gmm = init_skin_classifier(path)

# load in background replacement image
bg_img = cv2.imread('background.png')
bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

# setup webcam feed
cap = cv2.VideoCapture(0)

print('Opening webcam...')
while True:
    ret, frame = cap.read()

    if not ret: # if no frame is read
        break
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # create a region of interest for skin classifier using cv2 face detect
    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(frame_grey, 1.1, 4)
    print('Face detected' if len(face) > 0 else 'No Faces detected')

    eyes = eye_cascade.detectMultiScale(frame_grey, 1.1, 4)
    print('Eyes detected' if len(eyes) > 0 else 'No Eyes detected')
    
    print('Creating foreground mask with 3 elements...')
    fg_mask = np.zeros_like(frame[:, :, 0])    

    # if eyes were detected
    if len(eyes) > 0:
        # use detected eyes to generate mask for inner face region
        inner_face_mask = get_inner_face_mask(frame, eyes)

        # add new mask to foreground
        fg_mask = cv2.add(fg_mask, inner_face_mask)

    if len(face) > 0:
        for (x, y, w, h) in face:
            scale = 1.5
            center_x = x + (w/2)
            center_y = y + (h/2)
            w1 = int(scale * w)
            h1 = int(scale * h)
            x1 = int(center_x - ((w1)/2))
            y1 = int(center_y - ((h1)/2))
            skin_roi = (x1, x1+w1, y1, y1+h1) 

            # detect skin and create mask
            skin_mask = get_skin_mask(frame, skin_roi, skin_gmm, not_skin_gmm)
            print('[2/3] Skin mask done')

            # add to foreground mask
            fg_mask = cv2.add(fg_mask, skin_mask)


            # detect shoulders and create mask
            shoulder_mask = get_shoulder_mask(frame, x, y, w, h)
            print('[3/3] Shoulder mask done')

            # add to foreground mask
            fg_mask = cv2.add(fg_mask, shoulder_mask)


    # clip final foreground mask to ensure binary values
    fg_mask[fg_mask > 1] = 1

    # create background mask from foreground mask
    print('Isolating background...')
    bg_mask = cv2.bitwise_not(fg_mask)
    bg_mask[fg_mask >= 1] = 0

    # create final frame with mask 
    final_frame = cv2.bitwise_and(frame, frame, mask=fg_mask)
    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)

    # show frame with background replaced
    bg_img = cv2.resize(bg_img, (frame.shape[1], frame.shape[0]))
    bg_frame = cv2.bitwise_and(bg_img, bg_img, mask=bg_mask)
    frame_bg_replace = cv2.add(final_frame, bg_frame)
    cv2.imshow('Background Replacement', frame_bg_replace)

    # show frame with background blurred
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bg_blur = cv2.GaussianBlur(frame, (15, 15), sigmaX=8)
    bg_blur = cv2.bitwise_and(bg_blur, bg_blur, mask=bg_mask)
    frame_bg_blur = cv2.add(final_frame, bg_blur)
    cv2.imshow('Background Blurring', frame_bg_blur)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


