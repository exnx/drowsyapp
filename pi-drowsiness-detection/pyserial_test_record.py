import serial
import sys
import glob
import time
# from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import obd
import os

# define warning/alarm states
OKAY = 0
WARNING = 1
ALARM = 2
STATE = OKAY # 0: okay, 1: warning, 2: alarm
WARNING_COUNTER = 0
WARNING_THRESHOLD = 2 # number of warnings to allow before alarm instead

available_ports = []

ARDUINO_PORT = None
OBD_PORT = None
BUTTON_THRESHOLD = 1.5 # number of seconds to hold button to dismiss alarm
buttonPressStart = None
speed = 0

## -------------  For recording video


args = {
    "cascade": "haarcascade_frontalface_default.xml",
    "shape_predictor": "shape_predictor_68_face_landmarks.dat"
}
EYE_AR_THRESH = 0.2 # originally 0.3
EYE_AR_CONSEC_FRAMES = 16
lStart = None
lEnd = None
rStart = None
rEnd = None
detector = None
predictor = None
# VS = None
CAP = None
out_file = None

closeStart = None
openStart = None
CLOSE_THRESHOLD = 2 # number of seconds to allow eyes closed before warning/alarm
OPEN_THRESHOLD = 0.2 # number of seconds to confirm eyes open before dismissing close timer
ALARM_ON = False

# for calibrating camera window
x1,x2,y1,y2 = 400,900,750,1250

def get_available_serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    global available_ports
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')
    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass

    print("PORTS FOUND: {}".format(result))
    available_ports = result

def getPortName(connection):
    global available_ports
    if connection == "arduino":
        if '/dev/tty.Bluetooth-Incoming-Port' in available_ports: # running from MAC
            print("RUNNING FROM MACBOOK")
            return '/dev/tty.SLAB_USBtoUART'
        else: # running from RaspberryPi
            print("RUNNING FROM RASPBERRY PI")
            return '/dev/ttyUSB0'
    elif connection == "obd":
        if '/dev/tty.Bluetooth-Incoming-Port' in available_ports: # running from MAC
            print("RUNNING FROM MACBOOK")
            return '/dev/tty.usbserial-1420'
        else: # running from RaspberryPi
            print("RUNNING FROM RASPBERRY PI")
            return '/dev/ttyUSB1'

def checkForArduino(arduinoPortName='/dev/ttyUSB0'):
    global available_ports
    result = arduinoPortName in available_ports
    if result:
        print("FIND ARDUINO: SUCCESS")
    else:
        print("FIND ARDUINO: FAIL")
    return result

def connectToArduino(port='/dev/ttyUSB0'):
    global ARDUINO_PORT
    try:
        ARDUINO_PORT = serial.Serial(port, 9600)
        print("ARDUINO STATUS: CONNECTED")
    except(OSError, serial.SerialException):
        print("ARDUINO STATUS: NOT CONNECTED")

def initialize_Arduino():
    global STATE, OKAY, WARNING, ALARM, WARNING_COUNTER, buttonPressStart, available_ports
    portName = getPortName("arduino")
    if checkForArduino(portName):
        connectToArduino(portName)
        if ARDUINO_PORT:
            buttonPressStart = None
            ARDUINO_PORT.write(b'g')
    if ARDUINO_PORT:
        print("ARDUINO INITIALIZE: SUCCESS")
    else:
        print("ARDUINO INITIALIZE: FAIL")

def initialize_camera(verbose=False):
    global detector, predictor, lStart, lEnd, rStart, rEnd, CAP, out_file
    
    ## for writing a single video file, overwrites over and over
    # FILE_OUTPUT = 'output_{}.avi'.format(int(time.time()))
	# 
    # if os.path.exists(FILE_OUTPUT):
    #     os.remove(FILE_OUTPUT)
	
	# for writing multiple file outputs
    FILE_OUTPUT = ''
    # check for duplicate files, adds time stamp
    if os.path.exists('output.avi'):
        FILE_OUTPUT = 'output_{}.avi'.format(int(time.time()))
    else:
        FILE_OUTPUT = 'output.avi'
    if verbose: print("CAMERA INITIALIZE: Loading facial landmark predictor...")
    # load Haar cascade & create facial landmark predictor
    detector = cv2.CascadeClassifier(args["cascade"])
    predictor = dlib.shape_predictor(args["shape_predictor"])
    # get indices for left & right EYE_AR_THRESH
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    if verbose: print("CAMERA INITIALIZE: Starting video stream thread")
    # start video stream and wait temporarily
    # VS = VideoStream(src=0).start()
    CAP = cv2.VideoCapture(1)  # set camera number here (0 or 1)
    CAP.set(cv2.CAP_PROP_AUTOFOCUS, 0) # turn the autofocus off, unsure if works
    # width_val = int(CAP.get(3))
    # height_val = int(CAP.get(4))
    time.sleep(2.0)  # let camera warm up
	# Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
	# the video writer object
    # out_file = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(width_val),
    #     int(height_val)))
    out_file = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (int(x2-x1),
        int(y2-y1)))
    # vs = VideoStream(usePiCamera=True).start()
    if CAP:
        print("CAMERA INITIALIZE: SUCCESS")
    else:
        print("CAMERA INITIALIZE: FAIL")

def checkForOBD(OBDPortName='/dev/ttyUSB1'):
    global available_ports
    result = OBDPortName in available_ports
    if result:
        print("FIND OBD: SUCCESS")
    else:
        print("FIND OBD: FAIL")
    return result

def initialize_OBD():
    global OBD_PORT, available_ports
    portName = getPortName("obd")
    if checkForOBD(portName):
        OBD_PORT = obd.Async(portName, baudrate=115200)
        if OBD_PORT.status() == obd.OBDStatus.CAR_CONNECTED:
            print("OBD INITIALIZE: SUCCESS")
            OBD_PORT.watch(obd.commands.SPEED)
            OBD_PORT.start()
            print("OBD RUNNING: SUCCESS")
        else:
            print("OBD INITIALIZE: FAIL")

def handleArduino(verbose=False):
    global STATE, OKAY, WARNING, ALARM, buttonPressStart, WARNING_COUNTER
    if STATE == OKAY:
        ARDUINO_PORT.write(b'g')
    elif STATE == WARNING:
        print('WARNING DETECTED~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ {}'.format(WARNING_COUNTER))
        ARDUINO_PORT.write(b'y')
    elif STATE == ALARM:
        ARDUINO_PORT.write(b'r')
        data = ARDUINO_PORT.readline().rstrip()
        if data == b'button_press':
            buttonPressStart = time.time()
        elif data == b'button_release' and buttonPressStart is not None:
            duration = time.time() - buttonPressStart
            if duration > BUTTON_THRESHOLD:
                if verbose: print('HOLD LENGTH = {} sec. ALARM DISMISSED'.format(duration))
                STATE = OKAY
                WARNING_COUNTER = 0
                # TODO: stop alarm sound
            else:
                if verbose: print('HOLD LENGTH = {} sec. ALARM NOT DISMISSED'.format(duration))

def eye_aspect_ratio(eye):
	# compute the euclidean distances between the two sets of
	# vertical eye landmarks (x, y)-coordinates
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])

	# compute the euclidean distance between the horizontal
	# eye landmark (x, y)-coordinates
    C = np.linalg.norm(eye[0] - eye[3])

	# compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

	# return the eye aspect ratio
    return ear

def handleCamera(debugMode=False):
    global OBD, speed
    global out_file
    global VS, detector, predictor
    global lStart, lEnd, rStart, rEnd, EYE_AR_THRESH
    global CLOSE_THRESHOLD, OPEN_THRESHOLD
    global STATE, OKAY, WARNING, ALARM, closeStart, openStart, WARNING_COUNTER
    # grab frame, resize, and convert to grayscale
    # frame = VS.read()
    # frame = imutils.resize(frame, width=450)
    ret,frame = CAP.read()
    # width_val = int(CAP.get(3))
    # height_val = int(CAP.get(4))
    
    # print(width_val,height_val)
    # exit()
    frame = frame[x1:x2,y1:y2]  # height x width cropping
    # cv2.imshow("Frame",frame[0:50,0:50])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, 
        minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over all detected faces
    for (x,y,w,h) in faces:
        # construct a dlib rectable from Haar cascade bounding box
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # get facial landmarks as numpy array
        shape = face_utils.shape_to_np(predictor(gray, rect))
        # compute Eye Aspect Ratio (EAR) for both eyes & get average
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2
        # compute the convex hull for the left and right eye, then
         # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        if debugMode: 
            cv2.drawContours(frame, [leftEyeHull], -1,(0,255,0),1)
            cv2.drawContours(frame, [rightEyeHull], -1,(0,255,0),1)
        # check if average EAR is below threshold (eyes closed)
        if ear < EYE_AR_THRESH:
            # print("EAR: {}, THRESH: {}".format(averageEAR, EYE_AR_THRESH))
            if closeStart: # timer for eyes closed has started
                closeDuration = time.time() - closeStart
                openStart = None
                print("CLOSED DURATION: {} sec".format(closeDuration))
                if closeDuration > CLOSE_THRESHOLD: # eyes have been closed too long
                    print("DROWSINESS ALARM!")
                    if debugMode: cv2.putText(frame, "DROWSINESS ALARM!", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if STATE == OKAY and ((OBD_PORT and speed != 0) or (not OBD_PORT)): # if previous state was okay, trigger warning. if obd is connected, also check if car is moving
                        WARNING_COUNTER += 1
                        
                        if WARNING_COUNTER > WARNING_THRESHOLD:
                            STATE = ALARM
                            # Display Alarm on frame
                            if debugMode: cv2.putText(frame, "DROWSINESS ALARM!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        else:
                            STATE = WARNING
                            # Display warning on frame
                            if debugMode: cv2.putText(frame, "DROWSINESS WARNING!", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                closeStart = time.time() # start timer for eyes closed
        else:
            if STATE != ALARM:
                if openStart: # timer for eyes open has started
                    openDuration = time.time() - openStart
                    if openDuration > OPEN_THRESHOLD: # eyes have been open for long enough
                        print("OPEN DURATION: {} sec".format(openDuration))
                        closeStart = None # reset timer for eyes closed
                        if STATE == WARNING: # user "woke up"
                            STATE = OKAY
                else:
                    openStart = time.time()  # start timer for eyes closed
                    
        # draw the computed eye aspect ratio on the frame to help
		# with debugging and setting the correct eye aspect ratio
		# thresholds and frame counters
        if debugMode: cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if debugMode: 
		# show the frame
        cv2.imshow("Frame",frame)
        # recording the video to file
        debugMode: out_file.write(frame)
        key = cv2.waitKey(1) & 0xFF  # need this to allow frame to show up on screen
		# if the `q` key was pressed, break from the loop
        if key == ord("q"):
            pass

def handleOBD():
    global OBD_PORT, speed
    speed = OBD_PORT.query(obd.commands.SPEED).value.magnitude
    print("SPEED: {}".format(speed)) # non-blocking, returns immediately

def main(obd=True, arduino=True, debug=False):
    global STATE, OKAY, WARNING, ALARM
    i = 0
    handleCamera()
    while (True):
        if debug: 
            if not i % 100: print("STATE {}: {}".format(i, STATE))
        if CAP:
            handleCamera(debugMode=debug)
        else:
            print("ERROR: CAMERA DISCONNECTED")
            break
        if arduino:
            if ARDUINO_PORT:
                handleArduino()
            else:
                print("ERROR: ARDUINO DISCONNECTED")
                break
        if obd:
            if OBD_PORT.status() == obd.OBDStatus.CAR_CONNECTED:
                handleOBD()
            else:
                print("ERROR: OBD DISCONNECTED")
                break
        i += 1

def terminate():
    if CAP: CAP.release()
    if OBD_PORT and OBD_PORT.status() == obd.OBDStatus.CAR_CONNECTED: OBD_PORT.stop()
    print("PROGRAM TERMINATED")

if __name__ == "__main__":
    obd = False
    arduino = False
    debug = True
    STATE = OKAY
    WARNING_COUNTER = 0
    get_available_serial_ports()
    initialize_camera()
    if obd: initialize_OBD()
    if arduino: initialize_Arduino()
    main(obd=obd, arduino=arduino, debug=debug)
    terminate()
