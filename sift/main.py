import cv2
import numpy as np
import sys
import os

def get_sift_features(img, sift):
    kp, des = sift.detectAndCompute(img, None)
    return list(zip(kp, des))

def distance(d1, d2):
    return np.linalg.norm(d1 - d2)

img_dir = sys.argv[1]
query_dir = sys.argv[2]

img = cv2.imread(img_dir)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

query = cv2.imread(query_dir)
query = cv2.cvtColor(query, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()

img_features = get_sift_features(img, sift)
query_features = get_sift_features(query, sift)

query_kps = []
img_kps = []
for query_kp, query_des in query_features:
    distances = []
    for img_kp, img_des in img_features:
        distances.append((query_kp, img_kp, distance(query_des, img_des)))
    distances = sorted(distances, key=lambda x: x[2])
    query_kp, img_kp, d = distances[0]
    query_kps.append(query_kp)
    img_kps.append(img_kp)    

img = cv2.drawKeypoints(img, img_kps, img)
query = cv2.drawKeypoints(query, query_kps, query)
cv2.imshow('Result', img)
cv2.imshow('Query', query)
# cv2.imwrite('output/sift_{}'.format(os.path.basename(img_dir)),img)

cv2.waitKey(0)
cv2.destroyAllWindows()