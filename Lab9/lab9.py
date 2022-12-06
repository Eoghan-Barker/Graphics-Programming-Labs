import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy
import argparse
import random as rng

nrows = 3
ncols = 3


# # save image
# img = cv2.imread("ATU1.jpg",)
# img2 = cv2.imread('ATU2.jpg',cv2.IMREAD_GRAYSCALE)
# my image
img = cv2.imread("Otter_50.jpg",) 
img2 = cv2.imread('Otter2_50.jpg',cv2.IMREAD_GRAYSCALE)
imgFixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(nrows, ncols, 1), plt.imshow(imgFixed, cmap='gray')
plt.title("Original"), plt.xticks([]), plt.yticks([])

# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.subplot(nrows, ncols, 2), plt.imshow(gray, cmap='gray')
plt.title("Grayscale"), plt.xticks([]), plt.yticks([])

# Harris Corner detection
dst = cv2.cornerHarris(gray, 2, 3, 0.04)
imgHarris = copy.deepcopy(imgFixed)   # Deep copy of image
# loop through every element in the dst 2d matrix, 
# If the element is greater than a threshold draw a circle on the image
threshold = 0.99  # number between 0 and 1
for i in range(len(dst)):
    for j in range(len(dst[i])):
        if dst[i][j] > (threshold*dst.max()):
            cv2.circle(imgHarris, (j, i), 3, (100, 255, 0), -1)
plt.subplot(nrows, ncols, 3), plt.imshow(imgHarris, cmap='gray')
plt.title("Harris Corner Detection"), plt.xticks([]), plt.yticks([])


# Shi Tomasi Algorithm
corners = cv2.goodFeaturesToTrack(gray,30,0.01,10)
imgShiTomasi = copy.deepcopy(imgFixed)
# loop through the corners array and plot a circle at each corner
for i in corners:
    x,y = i.ravel()
    cv2.circle(imgShiTomasi,(int(x),int(y)),3,(255, 100, 0),-1)
plt.subplot(nrows, ncols, 4), plt.imshow(imgShiTomasi, cmap='gray')
plt.title("Shi Tomasi Algorithm"), plt.xticks([]), plt.yticks([])


# ORB detection
# Initiate ORB detector
orb = cv2.ORB_create()
# find the keypoints with ORB
kp = orb.detect(imgFixed,None)
# compute the descriptors with ORB
kp, des = orb.compute(imgFixed, kp)
# draw only keypoints location,not size and orientation
imgORB = cv2.drawKeypoints(imgFixed, kp, None, color=(0,255,0), flags=0)
plt.subplot(nrows, ncols, 5), plt.imshow(imgORB, cmap='gray')
plt.title("ORB"), plt.xticks([]), plt.yticks([])


# Feature Matching
# BruteForceMethod
# find the keypoints and descriptors with ORB
kp1, des1 = orb.detectAndCompute(gray,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)
# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)
# Draw first 10 matches.
BruteForceImg = cv2.drawMatches(gray,kp1,img2,kp2,matches[:10],None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
plt.subplot(nrows, ncols, 6), plt.imshow(BruteForceImg, cmap='gray')
plt.title("Brute Force Matching"), plt.xticks([]), plt.yticks([])

# FLANN Method
# Initiate SIFT detector
sift = cv2.SIFT_create()
# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(gray,None)
kp2, des2 = sift.detectAndCompute(img2,None)
# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params,search_params)
matches = flann.knnMatch(des1,des2,k=2)
# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]
draw_params = dict(matchColor = (0,255,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_DEFAULT)
imgFLANN = cv2.drawMatchesKnn(gray,kp1,img2,kp2,matches,None,**draw_params)
plt.subplot(nrows, ncols, 7), plt.imshow(imgFLANN, cmap='gray')
plt.title("FLANN Matching"), plt.xticks([]), plt.yticks([])


# Contour detection
rng.seed(12345)

threshold = 100
# Detect edges using Canny
canny_output = cv2.Canny(gray, threshold, threshold * 2)
# Find contours
contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Draw contours
drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
for i in range(len(contours)):
    color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
plt.subplot(nrows, ncols, 8), plt.imshow(drawing, cmap='gray')
plt.title("Contours"), plt.xticks([]), plt.yticks([])





# subplot the images with matplotlib
plt.show()