import convolution as conv
from utils import show_image
import cv2
import numpy as np
import sys

if __name__ == "__main__":
    image_dir = sys.argv[1].split(',')
    for img in image_dir:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)

def sift(image_dir):
    def get_keypoints(raw_img):
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        s = 3
        k = 2 ** (1.0 / s)
        sigma_0 = 1.6

        h, w = img.shape
        size_of_octave = s + 3
        height = size_of_octave - 1

        sigma = np.array([sigma_0 * k ** i for i in range(size_of_octave)])
        img_g = np.array([conv.filter(img, conv.get_gaussian_kernel(2*int(sigma[i]) + 1, sigma[i])) for i in range(size_of_octave)])
        dog = np.array([img_g[i + 1] - img_g[i] for i in range(height)])

        def is_extrema(pyramid, l, r, c):
            is_max, is_min = True, True
            for dl in range(-1, 2):
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dl == dr == dc == 0: continue
                        if pyramid[l, r, c] <= pyramid[l + dl, r + dr, c + dc]:
                            is_max = False
                        if pyramid[l, r, c] >= pyramid[l + dl, r + dr, c + dc]:
                            is_min = False
            return is_max or is_min

        extrema = np.zeros((height, h, w))
        for l in range(1, height - 1):
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    extrema[l, r, c] = is_extrema(dog, l, r, c)

        new_img = raw_img.copy()
        for l in range(height):
            X, Y = np.where(extrema[l])
            for x, y in zip(X, Y):
                cv2.circle(new_img, (y, x), 2*int(sigma[l]) + 1, (0, 0, 255))
        show_image(new_img, 'Extrema - %s' % image_dir)

        Ix = conv.filter(img, conv.sobel_h)
        Iy = conv.filter(img, conv.sobel_v)

        magnitude = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix)

        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy
            
        Sxx = conv.filter(Ixx, conv.get_gaussian_kernel(5, 1))
        Sxy = conv.filter(Ixy, conv.get_gaussian_kernel(5, 1))
        Syy = conv.filter(Iyy, conv.get_gaussian_kernel(5, 1))

        # Step 2: Harris response calculation
        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy
        r = det - 0.04*(trace**2)

        #Step 3: Find edges & corners using R

        thresh_corners = r > .005 * r.max()

        keypoints = []
        for l in range(height):
            R, C = np.where(np.logical_and(thresh_corners, extrema[l]))
            for r, c in zip(R, C):
                radius = int(sigma[l] + 0.5)
                if r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w: 
                    continue
                histogram = np.zeros(36)
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        angle = 180 + 180.0 / np.pi * theta[r + dr, c + dc]
                        bin_idx = np.clip(np.floor(angle / 10), 0, 35).astype(int)
                        histogram[bin_idx] += magnitude[r+dr,c+dc] / np.sum(magnitude)
                orientation,*_ = np.where(histogram == histogram.max())
                if len(orientation) > 2:
                    continue
                for ori in orientation:
                    size = magnitude[r, c] * 2e3 / np.sum(magnitude) * histogram[ori] * 1e3 / np.sum(histogram)
                    angle = ori * np.pi * 10.0/180 + 5
                    keypoints.append([r, c, sigma[l], angle])

        new_img = raw_img.copy()
        overlay = raw_img.copy()
        alpha = 0.5

        descriptors = np.zeros([len(keypoints), 128])
        out = []
        for i in range(len(keypoints)):
            r, c, s, a = keypoints[i]
            is_out = False
            for dr in range(-8, 8):
                for dc in range(-8, 8):
                    dc_rot = int(dc * np.cos(a) - dr * np.sin(a))
                    dr_rot = int(dc * np.sin(a) + dr * np.cos(a))

                    if not 0 <= r + dr_rot < h or not 0 <= c + dc_rot < w:
                        is_out = True
                        break
                    
                    angle = 180 + 180.0 / np.pi * theta[r + dr_rot, c + dc_rot] - a
                    if angle < 0: angle = 360 - angle
                    bin_idx = np.clip(np.floor((8.0 / 360) * angle), 0, 7).astype(int)
                    descriptors[i, 32 * int((dc + 8)/4) + 8 * int((dr + 8)/4) + bin_idx] += magnitude[r + dr_rot, c + dc_rot]

                    cv2.circle(overlay, (c + dc_rot, r + dr_rot), 1, (0, 0, 255, 0.1))
                    size = 2*int(s + 0.5 + 1) + 1
                    cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(a)), r+int(size * np.sin(a))), (255,0,0))
                if is_out: break
            if is_out: out.append(i)
        keypoints = [keypoints[i] for i in range(len(keypoints)) if i not in out]
        descriptors = [descriptors[i] for i in range(len(descriptors)) if i not in out]
        
        cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0, new_img)
        show_image(new_img, 'Histogram - %s' % image_dir)

        descriptors /= np.linalg.norm(descriptors)

        return keypoints, descriptors

    raw_img = cv2.imread(image_dir, cv2.IMREAD_COLOR)

    keypoints, descriptors = get_keypoints(raw_img)

    show_image(raw_img, 'Original - %s' % image_dir)

    new_img = raw_img.copy()
    for r, c, sigma, angle in keypoints:
        cv2.circle(new_img, (c, r), 2*int(sigma + 0.5) + 1, (0, 0, 255))
    show_image(new_img, 'Thresholded - %s' % image_dir)

    new_img = raw_img.copy()
    for r, c, sigma, angle in keypoints:
        cv2.circle(new_img, (c, r), 2*int(sigma + 0.5) + 1, (0, 0, 255))

        size = 2*int(sigma + 0.5 + 1) + 1
        cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(angle)), r+int(size * np.sin(angle))), (255,0,0))
    show_image(new_img, 'Orientation - %s' % image_dir)

    return keypoints, descriptors

if __name__ == "__main_":
    image_dir = sys.argv[1].split(',')

    for img in image_dir:
        sift(img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()