import serial
import sys
import glob
import time
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import serial
import sys
import glob
import time
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2
import obd
import os

machine = ""

# define warning/alarm states
OKAY = 0
WARNING = 1
ALARM = 2
STATE = OKAY # 0: okay, 1: warning, 2: alarm
WARNING_COUNTER = 0
WARNING_THRESHOLD = 3 # number of warnings to allow before alarm instead

available_ports = []

ARDUINO_PORT = None
OBD_PORT = None
BUTTON_THRESHOLD = 1.5 # number of seconds to hold button to dismiss alarm
buttonPressStart = None
speed = 0

args = {
    "cascade": "haarcascade_frontalface_default.xml",
    "shape_predictor": "shape_predictor_68_face_landmarks.dat"
}
EYE_AR_THRESH = 0.2 # originally 0.3
lStart = None
lEnd = None
rStart = None
rEnd = None
detector = None
predictor = None
VS = None

closeStart = None
openStart = None
CLOSE_THRESHOLD = 2 # number of seconds to allow eyes closed before warning/alarm
OPEN_THRESHOLD = 0.2 # number of seconds to confirm eyes open before dismissing close timer
ALARM_ON = False

FILE_OUTPUT = ''

# intialize the video writer object
out = cv2.VideoWriter()

# lets you record to a single file and overwrites (saving space)
def set_single_file():
    global FILE_OUTPUT
    FILE_OUTPUT = 'output.avi'
    if os.path.exists(FILE_OUTPUT):
        os.remove(FILE_OUTPUT)

# write multiple files with time stamps (takes more space)
def set_multi_file():
    global FILE_OUTPUT
    FILE_OUTPUT = 'output_{}.avi'.format(int(time.time()))

def get_available_serial_ports():
    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    global machine, available_ports
    if sys.platform.startswith('win'):
        machine = "windows"
        ports = ['COM%s' % (i + 1) for i in range(256)]
        print("RUNNING FROM WINDOWS")
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        machine = "linux"
        ports = glob.glob('/dev/tty[A-Za-z]*')
        print("RUNNING FROM RASPBERRY PI")
    elif sys.platform.startswith('darwin'):
        machine = "darwin"
        ports = glob.glob('/dev/tty.*')
        print("RUNNING FROM MACBOOK")
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
    global machine, available_ports
    if connection == "arduino":
        if machine == "darwin": # running from Macbook
            return '/dev/tty.SLAB_USBtoUART'
        elif machine == "linux": # running from RaspberryPi
            return '/dev/ttyUSB0'
    elif connection == "obd":
        if machine == "darwin": # running from Macbook
            return '/dev/tty.usbserial-1420'
        elif machine == "linux": # running from RaspberryPi
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

def initialize_camera(verbose=False, logCamera=True):
    global detector, predictor, lStart, lEnd, rStart, rEnd, VS, machine, out
    if verbose: print("CAMERA INITIALIZE: Loading facial landmark predictor...")
    # load Haar cascade & create facial landmark predictor
    detector = cv2.CascadeClassifier(args["cascade"])
    predictor = dlib.shape_predictor(args["shape_predictor"])
    # get indices for left & right EYE_AR_THRESH
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    if verbose: print("CAMERA INITIALIZE: Starting video stream thread")
    # start video stream and wait temporarily
    if machine=="linux" or not logCamera:
        VS = VideoStream(src=0).start()
    else:
        VS = VideoStream(src=1).start()
    # vs = VideoStream(usePiCamera=True).start()
    time.sleep(1.0)
    if VS:
        print("CAMERA INITIALIZE: SUCCESS")
    else:
        print("CAMERA INITIALIZE: FAIL")
    
    # get first frame, resize and get shape for outfile parameters
    frame = VS.read()
    frame = imutils.resize(frame, width=500)
    frame_height = frame.shape[0]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    
    # the video writer object
    out = cv2.VideoWriter(FILE_OUTPUT,fourcc, 20.0, (500,frame_height))

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

def handleCamera(verbose=False, display=False,recording=False):
    global VS, detector, predictor
    global lStart, lEnd, rStart, rEnd, EYE_AR_THRESH
    global CLOSE_THRESHOLD, OPEN_THRESHOLD
    global STATE, OKAY, WARNING, ALARM, closeStart, openStart, WARNING_COUNTER
    global out

    if verbose:
        if STATE == WARNING:
            print("WARNING WARNING WARNING WARNING WARNING WARNING {}".format(WARNING_COUNTER))
        elif STATE == ALARM:
            print("ALARM ALARM ALARM ALARM ALARM ALARM ALARM ALARM")

    # grab frame, resize, and convert to grayscale
    frame = VS.read()
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    # loop over all detected faces
    if len(faces) > 0:
        faceAreas = [w*h for (x,y,w,h) in faces]
        (x,y,w,h) = faces[np.argmax(faceAreas)] # bind x, y, w, and h to the largest face
        # construct a dlib rectable from Haar cascade bounding box
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        # get facial landmarks as numpy array
        shape = face_utils.shape_to_np(predictor(gray, rect))
        # compute Eye Aspect Ratio (EAR) for both eyes & get average
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        averageEAR = (leftEAR + rightEAR) / 2

        if display:
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # check if average EAR is below threshold (eyes closed)
        if averageEAR < EYE_AR_THRESH:
            # print("EAR: {}, THRESH: {}".format(averageEAR, EYE_AR_THRESH))
            if closeStart: # timer for eyes closed has started
                closeDuration = time.time() - closeStart
                openStart = None
                print("CLOSED DURATION: {} sec".format(closeDuration))
                if closeDuration > CLOSE_THRESHOLD: # eyes have been closed too long
                    if STATE == OKAY: # if previous state was okay, trigger warning
                        WARNING_COUNTER += 1
                        if WARNING_COUNTER > WARNING_THRESHOLD:
                            STATE = ALARM
                        else:
                            STATE = WARNING
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

        if display:
            cv2.putText(frame, "EAR: {:.3f}".format(averageEAR), (300, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
        if recording:
            out.write(frame)  # recording the video to file

def handleOBD():
    global OBD_PORT, speed
    speed = OBD_PORT.query(obd.commands.SPEED).value.magnitude
    print("SPEED: {}".format(speed)) # non-blocking, returns immediately

def main(verbose=False, display=False, use_arduino=True, use_obd=True,recording=False):
    global STATE, OKAY, WARNING, ALARM, machine

    if machine == "linux": display = False
    
    i = 0
    while (True):
        if not i % 100: print("STATE {}: {}".format(i, STATE))
        if VS:
            handleCamera(verbose=verbose, display=display,recording=recording)
        else:
            print("ERROR: CAMERA DISCONNECTED")
            break
        if use_arduino:
            if ARDUINO_PORT:
                handleArduino()
            else:
                print("ERROR: ARDUINO DISCONNECTED")
                break
        if use_obd:
            if OBD_PORT.status() == obd.OBDStatus.CAR_CONNECTED:
                handleOBD()
            else:
                print("ERROR: OBD DISCONNECTED")
                break
        i += 1

def terminate(display=False):
    if VS:
        if display: cv2.destroyAllWindows()
        VS.stop()
    # if OBD_PORT.status() == obd.OBDStatus.CAR_CONNECTED: OBD_PORT.stop()
    print("PROGRAM TERMINATED")

if __name__ == "__main__":
    STATE = OKAY
    WARNING_COUNTER = 0
    display = True
    use_arduino = True
    use_obd = False
    
    # choose either single or multi file recording
    recording = True
    record_single = False
    record_multi = True
    if record_single:
        set_single_file()
    if record_multi:
        set_multi_file()
    
    get_available_serial_ports()
    initialize_camera(logCamera=True)
    if use_obd: initialize_OBD()
    if use_arduino: initialize_Arduino()
    main(verbose=True, display=display, use_arduino=use_arduino, 
    use_obd=use_obd,recording=recording)
    terminate(display=display)
