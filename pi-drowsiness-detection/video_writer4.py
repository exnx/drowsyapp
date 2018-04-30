# For more info: http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_gui/py_video_display/py_video_display.html
import cv2
import numpy as np
import os
import time
import imutils
from imutils.video import VideoStream
from imutils.video import FPS

FILE_OUTPUT = ''

# check for duplicate files, adds time stamp
if os.path.exists('output.avi'):
    FILE_OUTPUT = 'output_{}.avi'.format(int(time.time()))
else:
    FILE_OUTPUT = 'output.avi'

# for recording
cap = cv2.VideoCapture(0)
width_val = int(cap.get(3))
height_val = int(cap.get(4))

print(height_val, width_val)

time.sleep(2.0)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# the video writer object
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width_val),int(height_val)))

while(cap.isOpened()):
	ret, frame = cap.read()
	if ret==True:
		frame = cv2.flip(frame,1)

        # write the flipped frame
		out.write(frame)

		cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	else:
		break
        
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
    
# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
