# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours

def returnPoints(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(gray)
	
	thresh = cv2.Canny(cl1, 50, 100)
	thresh = cv2.dilate(thresh, None, iterations=3)
	thresh = cv2.erode(thresh, None, iterations=3)
	cv2.bitwise_not ( thresh, thresh );

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	(cnts, _) = contours.sort_contours(cnts)
	pixelsPerMetric = None
	for c in cnts:
		# if the contour is not sufficiently large, ignore it
		if (cv2.contourArea(c) < 300 or cv2.contourArea(c) > 400):
			continue
 
		# compute the rotated bounding box of the contour
		'''orig = image.copy()
		box = cv2.minAreaRect(c)
		box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
		box = np.array(box, dtype="int")

		box = perspective.order_points(box)
		cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)
 
		# loop over the original points and draw them
		for (x, y) in box:
			cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)'''
		M = cv2.moments(c)
        	cX = int(M["m10"] / M["m00"]) if (M["m00"]!=0) else int(M["m10"])
       		cY = int(M["m01"] / M["m00"]) if (M["m00"]!=0) else int(M["m01"])
        	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		
		#return orig
	return image


def returnGray(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(gray)
	
	thresh = cv2.Canny(cl1, 50, 100)
	thresh = cv2.dilate(thresh, None, iterations=2)
	thresh = cv2.erode(thresh, None, iterations=2)
	cv2.bitwise_not ( thresh, thresh );

	return thresh
    
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True,help="path to output video file")
ap.add_argument("-p", "--picamera", type=int, default=-1,help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-f", "--fps", type=int, default=20,help="FPS of output video")
ap.add_argument("-c", "--codec", type=str, default="MJPG",help="codec of output video")
args = vars(ap.parse_args())

# initialize the video stream and allow the camera
# sensor to warmup
print("[INFO] warming up camera...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# initialize the FourCC, video writer, dimensions of the frame, and
# zeros array
fourcc = cv2.VideoWriter_fourcc(*args["codec"])
writer = None
(h, w) = (None, None)
zeros = None

# loop over frames from the video stream
while True:
	# grab the frame from the video stream and resize it to have a
	# maximum width of 300 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=300)

	# check if the writer is None
	if writer is None:
		# store the image dimensions, initialzie the video writer,
		# and construct the zeros array
		(h, w) = frame.shape[:2]
		zeros = np.zeros((h, w), dtype="uint8")
	output = np.zeros((h , w , 3), dtype="uint8")
	cv2.imshow("Frame", frame)
	cv2.imshow("Output", returnPoints(frame))
	cv2.imshow("Gray", returnGray(frame))
	key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break

# do a bit of cleanup
print("[INFO] cleaning up...")
cv2.destroyAllWindows()
vs.stop()
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
writer.release()