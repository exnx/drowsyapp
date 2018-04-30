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

# Checks and deletes the output file
# You cant have a existing file or it will through an error
# if os.path.isfile(FILE_OUTPUT):
#     os.remove(FILE_OUTPUT)

# Playing video from file:
# cap = cv2.VideoCapture('vtest.avi')
# Capturing video from webcam:
vs = VideoStream(src=1).start()
# frame = vs.read()
# frame = imutils.resize(frame, width=450)
# (height_val, width_val) = frame.shape[:2]

cap = cv2.VideoCapture(1)
width_val = int(cap.get(3))
height_val = int(cap.get(4))
# print(cap.shape)

print(height_val, width_val)

time.sleep(2.0)
fps = FPS().start()

# get the first one to get dimensions
# frame = vs.read()
# frame = imutils.resize(frame, width=400)
# (height_val, width_val) = frame.shape[:2]

 # to prevent duplicate images
currentFrame = 0

# Get current width of frame
# width_val = int(vs.get(3))
# width_val = vs.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)

# Get current height of frame
# height_val = int(vs.get(4))
# height_val = vs.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# the video writer object
out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width_val),int(height_val)))

while(True):
# while(cap.isOpened()):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    
    frame = vs.read()
	# frame = imutils.resize(frame, width=width_val)
    # frame = imutils.resize(frame, width=450)
    # frame = image(400,300,400,300)
    
    # Saves for video
    out.write(frame)
    
    # frame = frame[0:1500,0:1200]
    
    # Display the resulting frame
    cv2.imshow('frame',frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # To stop duplicate images
    currentFrame += 1
    
    fps.update()

# When everything done, release the capture
# cap.release()
vs.stop()
fps.stop()
out.release()
cv2.destroyAllWindows()
