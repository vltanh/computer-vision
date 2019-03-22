import cv2
import convolution as conv
import sys
import canny

if __name__ == "__main__":
    img_dir = sys.argv[1]

    img = cv2.imread(img_dir, 0)

    edge = canny.edge_detection(img, 
                                high_threshold_ratio=0.1, 
                                low_threshold_ratio=0.05,
                                visualize=False)

    cv2.waitKey(0)
    cv2.destroyAllWindows()