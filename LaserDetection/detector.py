import cv2
import numpy as np  

# load image
image_name = r"INSERT_IMAGE"
img = cv2.imread(image_name)

# convert to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# set lower and upper color limits
lower_val = np.array([0,0,235])
upper_val = np.array([255,25,255])

# Threshold the HSV image 
mask = cv2.inRange(hsv[750:2250,1000:3000], lower_val, upper_val)

# remove noise
kernel =  np.ones((5,5),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# find contours in mask
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours and polygon
for cnt in contours:
    cv2.drawContours(img[750:2250,1000:3000],[cnt],0,(0,0,255),2)
    cv2.fillPoly(img[750:2250,1000:3000], pts=[cnt],color=(0,0,255))

# show image
im = cv2.resize(img, (1000,750))
cv2.imshow("img", im)

# show mask
mask2 = cv2.resize(mask, (1000,750))
cv2.imshow("mask", mask2)

cv2.waitKey(0)
cv2.destroyAllWindows()
