import imutils
import cv2
import numpy as np
import time

start_time = time.time()


image = cv2.imread("./Images/template.png")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(gray)

cv2.imwrite('./Images/bnw.jpg',gray)

boundaries = [
	([210], [240]),
     ([105], [113]),
]

n=0

for (lower, upper) in boundaries:
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper,dtype="uint8")

    mask = cv2.inRange(gray, lower, upper)
    output = cv2.bitwise_and(gray, gray, mask = mask)

   
    thresh = cv2.threshold(output, 60, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    cv2.imwrite('./Images/thresh'+str(n)+'.jpg',thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE)
                        
    n+=1
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    for c in cnts:

        M = cv2.moments(c)
        cX = int(M["m10"] / M["m00"]) if (M["m00"]!=0) else int(M["m10"])
        cY = int(M["m01"] / M["m00"]) if (M["m00"]!=0) else int(M["m01"])
        cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        #cv2.circle(image, (cX, cY), 7, (255, 255, 255), -1)


print(time.time() - start_time)
cv2.imwrite('./Images/center.jpg',image)

