import cv2
import numpy as np
import sys
import os

img_dir = sys.argv[1]

img = cv2.imread(img_dir)
gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(gray,None)
cv2.drawKeypoints(gray, kp, img)

cv2.imwrite('output/sift_{}'.format(os.path.basename(img_dir)),img)