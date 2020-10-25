import numpy as np
import imutils
import cv2

# load the image
image = cv2.imread("finding_shapes_example.png", cv2.COLOR_BGR2HSV)
w, h = image.shape[:2]
image = cv2.resize(image, (h*2, w*2))
# find all the 'black' shapes in the image
lower = np.array([0, 0, 0])
upper = np.array([15, 15, 15])
shapeMask = cv2.inRange(image, lower, upper)

# find the contours in the mask
cnts = cv2.findContours(shapeMask.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
print("I found {} black shapes".format(len(cnts)))
cv2.imshow("Mask", shapeMask)
# loop over the contours
for c in cnts:
	# draw the contour and show it
	cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
cv2.imshow("Image", image)
cv2.waitKey(0)
