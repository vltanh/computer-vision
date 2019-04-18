import siftdetector
import cv2

img_dir = './input/query_2.jpg'
kps, descriptors = siftdetector.detect_keypoints(img_dir, 1)

img = cv2.imread(img_dir, cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
for x, y, _, _ in kps:
    x, y = int(x), int(y)
    cv2.circle(img, (y, x), 2, (0,0,255), -1)

cv2.imshow('Result', img)
cv2.imwrite('abc.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()