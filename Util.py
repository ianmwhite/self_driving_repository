# Name: Ian White
# Date: 5/3/24
# Honor Pledge: All work here is honestly obtained and is my own

import numpy as np
import cv2 as cv2

def getDeltaX(img):
    imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgF= imgGray.astype(np.float32)
    dX= np.zeros(np.shape(img),np.float32)
    dX= imgF[:, 1:len(imgF[0])]-imgF[:, 0:-1]
    return dX


def getXImage(img):
    dX = getDeltaX(img)
    dxN = normalize(dX)
    dxOut= dxN*255
    dxOut= dxOut.astype(np.uint8)
    # cv2.imshow("xDirection", dxOut)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return dxOut

def normalize(img):
    imgN= img.copy()
    minV= imgN.min()
    imgN -= minV
    maxV= imgN.max()
    imgN = imgN/ maxV
    return imgN

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


def createSketch(img):
    newImage, theta= getGradient(img)
    newImage= normalize(newImage)
    newImage= .9-newImage
    # cv2.imshow("Sketch Effect", newImage)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return newImage

def createOutline(img, color):
    imgParam= img.copy()
    newImage, theta= getGradient(img)
    newImage= normalize(newImage)

    lines= np.where(newImage>0.15)
    imgParam[lines]= color

    # cv2.imshow("outlineImage", imgParam)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imgParam

# Precond: sub has a subject against a green screen background
def greenScreen(sub, back):
    backCopy= back.copy()
    lower_hue = np.array([62, 50, 50])
    upper_hue = np.array([130, 255, 255])
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hue, upper_hue)

    points= np.where(mask==0)
    backCopy[points]= sub[points]


    # cv2.imshow("greenScreen", backCopy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return backCopy



def momentarilyDefinedGreenScreen(sub, back):
    backCopy= back.copy()
    # lower_hue = np.array([62, 50, 50]) #instead of hardcoding these values we will instead pick them based off the nth
    #camera frame that the camera sees
    # upper_hue = np.array([130, 255, 255])




    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hue, upper_hue)

    points= np.where(mask==0)
    backCopy[points]= sub[points]


    # cv2.imshow("greenScreen", backCopy)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return backCopy


# image1= cv2.imread("input/bobandlarry.png", 3)
# createSketch(image1)
#
# image2= cv2.imread("input/testpicture.png", 3)
# createOutline(image2, (0, 255, 0))

# image3 = cv2.imread("input/kitten_green.png", 3)
# image4= cv2.imread("input/wtqqnkYDYi2ifsWZVW2MT4-1200-80.png", 3)

# cv2.imshow("greenScreen", greenScreen(image3, image4))
# cv2.waitKey(0)
# cv2.destroyAllWindows()