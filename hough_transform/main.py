import cv2
import sys
import os
import numpy as np

def is_max(d, p, theta, r):
    for i in range(-r, r + 1):
        for j in range(-r, r + 1):
            v, _ = d.get((p+i, theta+j), [0, None])
            if d[(p, theta)][0] < v:
                return False
    return True


def non_max_suppression(d, r):
    non_max = []
    for (p, theta), _ in d.items():
        if not is_max(d, p, theta, r):
            non_max.append((p, theta))
    for k in non_max:
        d.pop(k)

if __name__ == "__main__":
    img_dir = sys.argv[1]

    img = cv2.imread(img_dir, 0)

    img_h, img_w = img.shape

    r, c = np.where(img == 255)
    edge_points = zip(c, r)
    
    accumulation_matrix = {}
    for x, y in edge_points:
        for theta in range(-90, 90+1, 2):
            _theta = theta * np.pi / 180.0
            p = int(x*np.cos(_theta) + y*np.sin(_theta))
            accumulation_matrix.setdefault((p, theta), [0,[]])
            accumulation_matrix[(p, theta)][0] += 1
            accumulation_matrix[(p, theta)][1].append((x, y))
    non_max_suppression(accumulation_matrix, 10)
    accumulation_matrix = [(k, v) for k, v in accumulation_matrix.items()]
    accumulation_matrix = sorted(accumulation_matrix, key=lambda x: x[1], reverse=True)

    cv2.imshow('Original', img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (p, theta), (cnt, points) in accumulation_matrix[:50]:
        for x,y in points:
            try:
                img[y][x] = [0, 255, 0]
            except:
                pass
        # _theta = theta * np.pi / 180.0
        # if theta == 0:
        #     x1 = x2 = p
        #     y1 = 0
        #     y2 = img_h
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0))
        # elif theta == -90 or theta == 90:
        #     y1 = y2 = int(p * 1.0 / np.sin(_theta))
        #     x1 = 0
        #     x2 = img_w
        #     cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255))
        # else:
        #     print(p, _theta)
        #     # p = xcos(theta) + ysin(theta)
        #     _theta = theta * np.pi / 180.0
        #     x1 = int(p * 1.0 / np.cos(_theta))
        #     y1 = 0
        #     y2 = img_w
        #     x2 = int((p - y2*np.sin(_theta)) * 1.0 / np.cos(_theta))
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0))

        #     x1 = img_h
        #     y1 = int((p - x1*np.cos(_theta)) * 1.0 / np.sin(_theta))
        #     x2 = 0
        #     y2 = int(p * 1.0 / np.sin(_theta))
        #     cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0))
        
    cv2.imshow('Line', img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()