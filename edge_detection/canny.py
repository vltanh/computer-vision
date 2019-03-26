import numpy as np
import convolution as conv
from utils import show_image
import itertools

NO_EDGE_VALUE = 0
WEAK_EDGE_VALUE = 64
STRONG_EDGE_VALUE = 255

def non_max_suppression(img, D):
    img_h, img_w = img.shape
    result = np.zeros_like(img)

    angle = D * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, img_h - 1):
        for j in range(1, img_w - 1):
            if (0 <= angle[i][j] < 22.5) or (157.5 <= angle[i][j] <= 180):
                q = img[i][j+1]
                r = img[i][j-1]
            elif (22.5 <= angle[i][j] < 67.5):
                q = img[i+1][j-1]
                r = img[i-1][j+1]
            elif (67.5 <= angle[i][j] < 112.5):
                q = img[i+1][j]
                r = img[i-1][j]
            elif (112.5 <= angle[i][j] < 157.5):
                q = img[i-1][j-1]
                r = img[i+1][j+1]

            if (img[i][j] >= q) and (img[i][j] >= r):
                result[i][j] = img[i][j]

    return result

def double_threshold(img, low_threshold_ratio=0.05, high_threshold_ratio=0.09):
    high_threshold = high_threshold_ratio * np.max(img)
    low_threshold = high_threshold * low_threshold_ratio

    result = np.zeros_like(img)

    strong_i, strong_j = np.where(img >= high_threshold)
    weak_i, weak_j = np.where((low_threshold <= img) & (img < high_threshold))

    result[strong_i,strong_j] = STRONG_EDGE_VALUE
    result[weak_i,weak_j] = WEAK_EDGE_VALUE

    return result

def is_strong_edge(img, r, c):
    dr = (-1, -1, -1, 0, 0, 1, 1, 1)
    dc = (-1, 0, 1, -1, 1, -1, 0, 1)
    for i in range(8):
        if img[r+dr[i]][c + dc[i]] == STRONG_EDGE_VALUE:
            return True
    return False

def hysteresis(img):
    img_h, img_w = img.shape
    result = img.copy()
    for r in range(1, img_h - 1):
        for c in range(1, img_w - 1):
            if img[r][c] == WEAK_EDGE_VALUE:
                result[r][c] = STRONG_EDGE_VALUE if is_strong_edge(img, r, c) else NO_EDGE_VALUE
    return result

def edge_detection(img, high_threshold_ratio=0.09, low_threshold_ratio=0.05, visualize=True):
    # Step 1: Blur
    blurred = conv.convolution(img, conv.gaussian_55_blur)

    # Step 2: Edge detection
    dx = conv.convolution(blurred, conv.sobel_h)
    dy = conv.convolution(blurred, conv.sobel_v)

    d = np.sqrt(dx*dx + dy*dy).astype('int')
    angle = np.arctan2(dy, dx)

    # Step 3: Non-maximum suppression
    nomax = non_max_suppression(d, angle)

    # Step 4: Hysteresis
    thresholded = double_threshold(nomax, high_threshold_ratio=high_threshold_ratio, 
                                            low_threshold_ratio=low_threshold_ratio)
    final = hysteresis(thresholded)
    for _ in range(100):
        final = hysteresis(final)

    # Visualization
    if visualize:
        show_image(img, 'Original')
        show_image(blurred, 'Blur')
        show_image(dx, 'Horizontal Gradient')
        show_image(dy, 'Vertical Gradient')
        show_image(d, 'Gradient Intensity')
        show_image(nomax, 'Non-max Suppression')
        show_image(thresholded, 'Double Threshold')
        show_image(final, 'Hysteresis')

    return final