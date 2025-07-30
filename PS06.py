
# Name: Ian White
# Date: 4/17/24

# Honor Pledge: All work here is honestly obtained and is my own.

#### Needed Import Statements ####
import numpy as np
import cv2 as cv2


#### Ringbuffer that holds the last 12 vanishing points
averageVals= [[],
                []]


#### Complete Questions Below ####

img1= cv2.imread("input/sample1.png")
img2= cv2.imread("input/sample2.png")
img3= cv2.imread("input/sample3.png")
img4= cv2.imread("input/sample4.png")
img5= cv2.imread("input/sample5.png")






# Question 1
img1_g= cv2.imread("input/sample1.png", 0)
img1_edge= cv2.Canny(img1_g, 50, 200, None, 3)
cv2.imwrite("output/img1_edge.png", img1_edge)

lines= cv2.HoughLines(img1_edge, 1, np.pi/180, 150, None, 0, 0)



# Function to draw lines
def drawLines(img, lines):
    for i in range(len(lines)):
        rho= lines[i][0][0]
        theta= lines[i][0][1]

        dx= np.cos(theta)
        dy= np.sin(theta)



        x0= dx* rho
        y0= dy* rho

        pt1= (int(x0 + 1000*(-dy))), (int(y0+ 1000*(dx)))
        pt2 = (int(x0- 1000 * (-dy))), (int(y0 - 1000 * (dx)))

        cv2.line(img, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
        # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.LINE_AA )
    return img

def drawFootballLines(img, lines):
    for i in range(len(lines)):
        rho= lines[i][0][0]
        theta= lines[i][0][1]

        print("Rho: ", rho, "| Theta: ", theta)

        if(theta<1.55 or theta>1.6):

            dx= np.cos(theta)
            dy= np.sin(theta)



            x0= dx* rho
            y0= dy* rho

            pt1= (int(x0 + 1000*(-dy))), (int(y0+ 1000*(dx)))
            pt2 = (int(x0- 1000 * (-dy))), (int(y0 - 1000 * (dx)))

            cv2.line(img, pt1, pt2, (0,255,0), 1, cv2.LINE_AA)
            # cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), cv2.LINE_AA )
    return img


# My guess is the center is 320, 240
def drawDot(img, x, y):
    cv2.circle(img, (x,y), 10, (0,0,255), -1)

    if(x<300):
        print("Turn Left")
    elif(x>350):
        print("Turn Right")


# Takes an array of the last 12 measurements of the findVanishingPoints
# and averages them to draw the vanishing point dot
def drawAverageDot(img, averageArray):
    xS= []
    yS= []

    for i in averageArray[0]:
        xS.append(i)

    for pen15 in averageArray[1]:
        yS.append(pen15)

    xVal= (int) (sum(xS)/len(averageArray[0]))
    yVal= (int) (sum(yS)/len(averageArray[1]))
    cv2.circle(img, (xVal, yVal), 10, (255, 255, 255), -1)

    if (xVal < 300):
        print("Turn Left")
    elif (xVal>350):
        print("Turn Right")


# Question 2
# img2_g= img2.copy()
# img2_g= img2_g[:, :, 0:1]
# img2_noise= cv2.GaussianBlur(img2_g, (15,15), 2)
# # cv2.imwrite("output/img2_noise.png", img2_noise)
# img2_edge= cv2.Canny(img2_noise, 50, 150)
# cv2.HoughLines(img1_edge, 1, np.pi/180, 150, None, 0, 0)
# img2_lines= drawLines(img2, lines)
# cv2.imwrite("output/img2_lines.png", img2_lines)
# cv2.imshow("lines", img2_lines)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


def findLines(imgParam, threshold, reactionTime):
    # Reduce image noise
    img2_g = imgParam.copy()
    img2_g = img2_g[:, :, 0:1]
    img2_noise = cv2.GaussianBlur(img2_g, (15, 15), 2)


    # cv2.imwrite("output/img2_noise.png", img2_noise)
    img2_edge = cv2.Canny(img2_g, 50, 150)
    # cv2.imshow("edges", img2_edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    lines= cv2.HoughLines(img2_edge, 1, np.pi / 180, threshold, None, 0, 0)
    if (lines is not None):
        if(len(averageVals[0])<reactionTime):
            (x, y) = getVanishingPoint(lines, imgParam)
            averageVals[0].append(x)
            averageVals[1].append(y)
            # print(lines)
            # img2_lines = drawLines(imgParam, lines)
            img2_lines = drawDot(imgParam, x, y)
            # cv2.imwrite("output/foundLines.png", img2_lines)
            # cv2.imshow("lines", img2_lines)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
        else:
            (x, y) = getVanishingPoint(lines, imgParam)
            averageVals[0][:len(averageVals[0])-1]= averageVals[0][1: reactionTime]
            averageVals[0][reactionTime-1]= x
            averageVals[1][:len(averageVals[1])-1]= averageVals[1][1:reactionTime]
            averageVals[1][reactionTime-1]= y
            drawAverageDot(imgParam, averageVals)
            drawDot(imgParam, x, y)
            cv2.line(imgParam, (320, 0), (320, 480), (255, 0, 0), 3)
            cv2.line(imgParam, (0, 240), (640, 240), (255, 0, 0), 3)
        return imgParam
    else:
        return imgParam

def findFootballLines(imgParam, threshold):
    img2_g = imgParam.copy()
    img2_g = img2_g[:, :, 0:1]
    img2_noise = cv2.GaussianBlur(img2_g, (15, 15), 2)
    # cv2.imwrite("output/img2_noise.png", img2_noise)
    img2_edge = cv2.Canny(img2_g, 50, 150)
    cv2.imshow("edges", img2_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    lines= cv2.HoughLines(img2_edge, 1, np.pi / 180, threshold, None, 0, 0)

    img2_lines = drawFootballLines(imgParam, lines)

    # cv2.imwrite("output/foundLines.png", img2_lines)
    # cv2.imshow("lines", img2_lines)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return imgParam

#returns b, m
def getVanishingPoint(lines, img):
    AMatrix = []
    BMatrix = []
    for i in range(len(lines)):
        rho= lines[i][0][0]
        theta= lines[i][0][1]
        dx= np.cos(theta)
        dy= np.sin(theta)



        x0= dx* rho
        y0= dy* rho

        pt1= (int(x0 + 1000*(-dy))), (int(y0+ 1000*(dx)))
        pt2 = (int(x0- 1000 * (-dy))), (int(y0 - 1000 * (dx)))

        YS= pt2[1]-pt1[1]
        XS= pt2[0]-pt1[0]

        if(XS==0):
            m=100000
        else:
            m= (YS/XS)


        if(abs(m)>.5 and abs(m)<7):
            b= pt1[1]- (m*pt1[0])
            BMatrix.append([b])
            AMatrix.append([m,1])
            cv2.line(img, pt1, pt2, (0, 255, 0), 1, cv2.LINE_AA)
            # print(AMatrix)
    if(len(AMatrix)>1 and len(BMatrix)>1):
        X= np.linalg.lstsq(AMatrix, BMatrix, rcond= False)

        '''Returns x, y'''
        xV= -(int)(X[0][0][0])
        yV = (int)(X[0][1][0])
        return (xV, yV)
    else:
        if(len(averageVals[0])>1):
            return (averageVals[0][len(averageVals[0])-1], averageVals[1][len(averageVals[1])-1])
        else:
            return 125, 125


# img1_lines= drawLines(img1, lines)
# cv2.imshow("lines", img1_lines)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# imgPrint= findLines(img3, 100)
# cv2.imwrite("output/foundLines3.png", imgPrint)
# cv2.imshow("lines", imgPrint)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# imgPrint= findLines(img4, 200)
# # cv2.imwrite("output/foundLines4.png", imgPrint)
# cv2.imshow("lines", imgPrint)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# imgPrint= findFootballLines(img5, 150)
# cv2.imwrite("output/img_5_yardLines.png", imgPrint)
# cv2.imshow("lines", imgPrint)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# Camera code
cap = cv2.VideoCapture(0)

# Camera Loop - Read image from camera and display
while (True):
    ret, frame = cap.read()

    # Show Video
    if ret==True:
        cv2.imshow('Camera', frame)
        cv2.imshow('Lines', findLines(frame, 100, 20))
    # Exit Sequence
    # Exits on 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# Release cap object from memory and turn off camera
cap.release()


