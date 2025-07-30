import cv2 as cv2
import numpy as np
import robotpy_apriltag as apriltag

options= apriltag.AprilTagDetection(families= "tag36h11")
detector= apriltag.AprilTagDetector(options)

