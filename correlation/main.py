import cv2
import correlation as cor
import sys
import os

def non_max(img, r, c):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if img[r + i][c + j] > img[r][c]:
                return False
    return True

if __name__ == "__main__":
    img_dir = sys.argv[1]
    kernel_dir = sys.argv[2]

    img = cv2.imread(img_dir, 0)
    kernel = cv2.imread(kernel_dir, 0)

    img = cv2.bitwise_not(img).astype('int')
    kernel = cv2.bitwise_not(kernel).astype('int')

    response_map = cor.correlation(img, kernel, visualize=False)
    h, w = response_map.shape
    for r in range(1, h-1):
        for c in range(1, w-1):
            if non_max