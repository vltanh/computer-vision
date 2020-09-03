import cv2
import convolution as conv
import sys
import canny
import os
import numpy as np
from utils import transform_visualizable, show_image

def detect_by_canny(img, high_threshold_ratio, low_threshold_ratio, visualize=True):
    return canny.edge_detection(img, 
                                high_threshold_ratio=high_threshold_ratio, 
                                low_threshold_ratio=low_threshold_ratio,
                                visualize=visualize)

def detect_by_sobel(img, visualize=True):
    dx = conv.convolution(img, conv.sobel_h)
    dy = conv.convolution(img, conv.sobel_v)
    d = np.sqrt(dx*dx + dy*dy).astype('int')
    return transform_visualizable(d)

def detect_by_prewitt(img, visualize=True):
    dx = conv.convolution(img, conv.prewitt_h)
    dy = conv.convolution(img, conv.prewitt_v)
    d = np.sqrt(dx*dx + dy*dy).astype('int')
    return transform_visualizable(d)

def detect_by_laplacian(img, visualize=True):
    return transform_visualizable(conv.convolution(img, conv.laplacian_8))

CMD_SOBEL = 'sobel'
CMD_PREWITT = 'prewitt'
CMD_LAPLACIAN = 'laplace'
CMD_CANNY = 'canny'

if __name__ == "__main__":
    img_dir = sys.argv[1]
    cmd = sys.argv[2]

    img = cv2.imread(img_dir, 0)
    
    edge = None
    if cmd == CMD_SOBEL:
        edge = detect_by_sobel(img)
    elif cmd == CMD_PREWITT:
        edge = detect_by_prewitt(img)
    elif cmd == CMD_LAPLACIAN:
        edge = detect_by_laplacian(img)
    elif cmd == CMD_CANNY:
        high_thres = float(sys.argv[3])
        low_thres = float(sys.argv[4])
        edge = detect_by_canny(img, high_thres, low_thres)
    
    cv2.imshow('Result', edge)
    cv2.imwrite('output/edge_{}_{}'.format(cmd, os.path.basename(img_dir)), edge)

    cv2.waitKey(0)
    cv2.destroyAllWindows()