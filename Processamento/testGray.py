import imutils
import cv2
import numpy as np

image = cv2.imread("template.png")
cv2.imshow("Original", image)

boundaries = [
	([0, 20, 255], [50, 80, 255]),
]

for (lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper,dtype="uint8")

    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)

    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]


    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)

    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    for c in cnts:

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"]) if (M["m00"]!=0) else int(M["m10"])
        cY = int(M["m01"] / M["m00"]) if (M["m00"]!=0) else int(M["m01"])
 
        	# draw the contour and center of the shape on the image
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)
        #cv2.putText(image, "center", (cX - 20, cY - 20),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.imshow("Image", image)
    cv2.waitKey(0)

cv2.destroyAllWindows()
