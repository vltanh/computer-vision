import cv2
import correlation as cor
import sys
import os
from utils import show_image

def is_max(img, r, c):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if img[r + i][c + j] > img[r][c]:
                return False
    return True

if __name__ == "__main__":
    img_dir = sys.argv[1]
    kernel_dir = sys.argv[2]
    thres = float(sys.argv[3])

    img = cv2.imread(img_dir, 0)
    kernel = cv2.imread(kernel_dir, 0)

    img = cv2.bitwise_not(img).astype('int')
    kernel = cv2.bitwise_not(kernel).astype('int')

    response_map = cor.correlation(img, kernel, visualize=False)
    h, w = response_map.shape
    thres = response_map.max() * thres
    response = []
    for r in range(1, h-1):
        for c in range(1, w-1):
            if not response_map[r][c] > thres:
                response_map[r][c] = 0
            else:
                response.append((r, c))
    show_image(response_map, 'Response map')

    img = cv2.cvtColor(cv2.bitwise_not(img).astype('uint8'), cv2.COLOR_GRAY2BGR)
    for r, c in response:
        cv2.circle(img, (c, r), 10, (0,0,255), 2)

    show_image(img, 'Result')

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    