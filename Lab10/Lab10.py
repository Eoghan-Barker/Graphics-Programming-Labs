import cv2
import numpy as np
from matplotlib import pyplot as plt

# import xml files for pre-trained face, eye and smile detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# save image and convert to gray
imgBGR = cv2.imread("smileandfrown.jpg",)
img = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(imgBGR, cv2.COLOR_BGR2GRAY)

# if faces are found it will draw a rectangle around them, then check for eye and smile detection and draw rectangles around them
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x,y,w,h) in faces:
    # draw red rectangle around face
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        # draw green rectangle around eyes
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    smiles = smile_cascade.detectMultiScale(roi_gray)
    for (sx,sy,sw,sh) in smiles:
        # draw green rectangle around smiles
        cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)

# split the image into r, g, b color channels
img2 = cv2.imread("twopeople.jpg")
img2Fixed = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

# split the Blue, Green and Red color channels
blue,green,red = cv2.split(img2)

# define channel having all zeros
zeros = np.zeros(blue.shape, np.uint8)

# merge zeros to make BGR image
blueBGR = cv2.merge([blue,zeros,zeros])
greenBGR = cv2.merge([zeros,green,zeros])
redBGR = cv2.merge([zeros,zeros,red])

# convert image to HSV
img2HSV = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(img2HSV)



# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# subplot the images with matplotlib
nrows = 3
ncols = 3
plt.subplot(nrows, ncols, 1), plt.imshow(img, cmap='gray')
plt.title("Face/Eye/Smile Detection"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(img2Fixed, cmap='gray')
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(blueBGR, cmap='gray')
plt.title("Blue Channel"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 4), plt.imshow(greenBGR, cmap='gray')
plt.title("Green Channel"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 5), plt.imshow(redBGR, cmap='gray')
plt.title("Red Channel"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 6), plt.imshow(h, cmap='gray')
plt.title("Hue Channel"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 7), plt.imshow(s, cmap='gray')
plt.title("Saturation Channel"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 8), plt.imshow(v, cmap='gray')
plt.title("Value Channel"), plt.xticks([]), plt.yticks([])
plt.show()





