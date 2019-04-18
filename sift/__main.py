import cv2
import numpy as np
import sys
import os

def get_sift_features(img, sift):
    kp, des = sift.detectAndCompute(img, threshold=1.5)
    return list(zip(kp, des))

def distance(d1, d2):
    return np.linalg.norm(d1 - d2)

img_dir = sys.argv[1]
img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create(100)

kp, des = sift.detectAndCompute(img, None)

img = cv2.drawKeypoints(img, kp, img)
cv2.imshow('Result', img)

cv2.waitKey(0)
cv2.destroyAllWindows()