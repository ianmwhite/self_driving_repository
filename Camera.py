# Camera Sample in Python
# Save as a .py file
# Mr. Michaud
# www.nebomusic.net

import cv2 as cv2
import numpy as np


import PS06
# Setup Camera Object: 0 is the index of the installed camera
# 1 is the index of a USB camera (webcam)

cap = cv2.VideoCapture(0)

# Camera Loop - Read image from camera and display
while (True):
    ret, frame = cap.read()

    # Show Video
    if ret==True:
        cv2.imshow('Camera', frame)
        cv2.imshow('Lines', PS06.findLines(frame, 100, 20))
    # Exit Sequence
    # Exits on 'q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break

# Release cap object from memory and turn off camera
cap.release()


