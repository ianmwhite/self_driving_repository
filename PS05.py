# Name: Ian White
# Date: 4/8/24

# Honor Pledge:
# All work here is honestly obtained and is my own

import numpy as np
import cv2 as cv2

img1= cv2.imread("Input5/boringBird.png", 3)
img_noise= cv2.GaussianBlur(img1, (15,15), 2)


# Question 3
'''
Implement functions to map the gradient distance in each pixel. You may use the functions from the
previous problem set. (You may copy and paste these functions over from PS04).
'''
def getDeltaX(img):
    imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgF= imgGray.astype(np.float32)
    dX= np.zeros(np.shape(img),np.float32)
    dX= imgF[:, 1:len(imgF[0])]-imgF[:, 0:-1]
    return dX


def getDeltaY(img):
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgF = imgGray.astype(np.float32)
    dY= np.zeros(np.shape(img),np.float32)
    dY= imgF[1:len(imgF), : ]-imgF[0:-1, : ]
    # dY= np.transpose(dY)
    return dY

def getGradient(img):
    xDistance= getDeltaX(img)
    xDistance= xDistance[:-1, :]
    yDistance= getDeltaY(img)
    yDistance= yDistance[:, :-1]
    # print("xDistance: ", np.square(xDistance))
    # print("yDistance: ", np.square(yDistance))
    magnitude= np.sqrt(np.square(xDistance)+ np.square(yDistance))
    theta= np.arctan2(yDistance, xDistance)

    return magnitude, theta
# Precondition: imgN is already a float32
def normalize(img):
    imgN= img.copy()
    minV= imgN.min()
    imgN -= minV
    maxV= imgN.max()
    imgN = imgN/ maxV
    return imgN

testGrid= np.array([[4,4],
                    [5,7]])

def getXImage(img):
    dX = getDeltaX(img)
    dxN = normalize(dX)
    dxOut= dxN*255
    dxOut= dxOut.astype(np.uint8)
    # cv2.imshow("xDirection", dxOut)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dxOut


def getYImage(img):
    dY = getDeltaY(img_noise)
    dyN = normalize(dY)
    dyOut = dyN * 255
    dyOut = dyOut.astype(np.uint8)
    return dyOut



def getEdgeDetection(img, threshold):
    edgeImage= np.zeros(img.shape, np.float32)
    D, T= getGradient(img)
    DN= normalize(D)


    register= np.where(D>threshold)
    edgeImage[register]= 255
    return edgeImage


# print(getDeltaX(img_noise))
# print(getDeltaY(img_noise))
# print(getGradient(img_noise))


# getXImage(img_noise)

edges= getEdgeDetection(img_noise, 5)
cv2.imshow("edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("Output/edges.png", edges)

cv2.imwrite("Output/xImage.png", getXImage(img_noise))
cv2.imwrite("Output/yImage.png", getYImage(img_noise))
cv2.imshow("Output", edges)