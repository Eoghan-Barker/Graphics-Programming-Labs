import cv2
import numpy as np
from matplotlib import pyplot as plt
import copy

nrows = 3
ncols = 4


# save image
img = cv2.imread("ATU.jpg",)
imgFixed = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# show images
# cv2.imshow('Original image',img)
# cv2.imshow('Gray image', gray)
# # wait for user input
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Blur the image
imgBlur3x3 = cv2.GaussianBlur(gray, (3, 3), 0)
imgBlur13x13 = cv2.GaussianBlur(gray, (13, 13), 0)

# Use sobel operator to perform edge detection
sobelHorizontal = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)  # x dir
sobelVertical = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)  # y dir
sobelSum = cv2.addWeighted(sobelHorizontal, 0.5, sobelVertical, 0.5, 0)

# Canny Edge Detection
canny = cv2.Canny(gray, 100, 200)

# import my own image
myImg = cv2.imread("mypic.jpeg",)
myGray = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)
myCanny = cv2.Canny(myGray, 150, 300)

# create a threshold for the sobelSum img and set all pixels above the threshold to 1 and below to 0.
sobelThreshold = copy.deepcopy(sobelSum)
# loop through all the pixel values
for i in range(0, sobelThreshold.shape[0]) :
    for j in range(0, sobelThreshold.shape[1]) :
        pixel = sobelThreshold[i][j]
        # compare pixel value to threshold and add it as black or white to the image
        if pixel < 500:
            sobelThreshold.itemset(i, j, 0)
        else:
            sobelThreshold.itemset(i, j, 1)


# manual edge detection
edgeImg = copy.deepcopy(gray)
# iterate over the image
for i in range(0, edgeImg.shape[0]):
    for j in range(0, edgeImg.shape[1]):
        # save pixel to compare to neighboring pixels
        horizontalCheck = edgeImg.item(i, j)
        # make sure comparing to pixels in bounds
        if i < gray.shape[0] - 1:
            horizontalNeighbor = gray.item(i + 1, j)
        else:
            horizontalNeighbor = horizontalCheck

        # do the same vertically
        verticalCheck = gray.item(i, j)
        if j < gray.shape[1] - 1:
            verticalNeighbor = gray.item(i, j + 1)
        else:
            verticalNeighbor = verticalCheck

        # compare the pixels and add edges to image
        edgeImg.itemset(i, j, np.abs((horizontalCheck - horizontalNeighbor) + (verticalCheck - verticalNeighbor)))
        




# subplot the images with matplotlib
plt.subplot(nrows, ncols, 1), plt.imshow(imgFixed, cmap='gray')
plt.title("Original"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 2), plt.imshow(gray, cmap='gray')
plt.title("GrayScale"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 3), plt.imshow(imgBlur3x3, cmap='gray')
plt.title("3x3 Blur"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 4), plt.imshow(imgBlur13x13, cmap='gray')
plt.title("13x13 Blur"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 5), plt.imshow(sobelHorizontal, cmap='gray')
plt.title("Sobel Horizontal"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 6), plt.imshow(sobelVertical, cmap='gray')
plt.title("Sobel Vertical"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 7), plt.imshow(sobelSum, cmap='gray')
plt.title("Sobel Sum"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 8), plt.imshow(canny, cmap='gray')
plt.title("Canny Edge Image"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 9), plt.imshow(myCanny, cmap='gray')
plt.title("Canny Edge myImage"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 10), plt.imshow(sobelThreshold, cmap="gray")
plt.title("Sobel Range"), plt.xticks([]), plt.yticks([])
plt.subplot(nrows, ncols, 11), plt.imshow(edgeImg, cmap="gray")
plt.title("Manual Sobel"), plt.xticks([]), plt.yticks([])
plt.show()
