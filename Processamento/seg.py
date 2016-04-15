# -*- coding: utf-8 -*-

import cv2
import numpy as np

image = cv2.imread("megaman_8bits.png")

boundaries = [
	([245, 180, 47], [255, 190, 57]),
     ([229, 103, 0], [239, 113, 10]),
     ([245, 245, 245], [255, 255, 255]),
     ([155, 221, 245], [165, 231, 255]),
     ([0, 0, 0], [10, 10, 10]),
]

cv2.imshow("Original", image)

for (lower, upper) in boundaries:
	# create NumPy arrays from the boundaries
    lower = np.array(lower, dtype = "uint8")
    upper = np.array(upper,dtype="uint8")
	# find the colors within the specified boundaries and apply
	# the mask
    mask = cv2.inRange(image, lower, upper)
    output = cv2.bitwise_and(image, image, mask = mask)
 
    # show the images
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("Seg", thresh)
    
    cv2.waitKey(0) 
cv2.destroyAllWindows()