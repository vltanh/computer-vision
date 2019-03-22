import cv2
import convolution as conv
import sys
import canny
import os

if __name__ == "__main__":
    img_dir = sys.argv[1]
    high_thres = float(sys.argv[2])
    low_thres = float(sys.argv[3])

    img = cv2.imread(img_dir, 0)

    edge = canny.edge_detection(img, 
                                high_threshold_ratio=high_thres, 
                                low_threshold_ratio=low_thres,
                                visualize=True)
    
    cv2.imwrite('output/edge_{}_{}_{}'.format(high_thres, low_thres, os.path.basename(img_dir)), edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()