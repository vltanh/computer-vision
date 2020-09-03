import convolution as conv
from utils import show_image, transform_visualizable
import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

def sift(image_dir, raw_img, feature_extractor):
    def get_keypoints_harris(raw_img):
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape

        Ix = conv.filter(img, conv.sobel_h)
        Iy = conv.filter(img, conv.sobel_v)

        magnitude = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix)

        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy

        window_size = 3
        kernel = conv.get_gaussian_kernel(window_size, 1)
        Sxx = conv.filter(Ixx, kernel)
        Sxy = conv.filter(Ixy, kernel)
        Syy = conv.filter(Iyy, kernel)

        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy
        r = det - 0.04*(trace**2)

        thresh_corners = r > .05 * np.max(r)

        # show_image(thresh_corners*255, 'corners')

        keypoints = []
        R, C = np.where(thresh_corners)
        for r, c in zip(R, C):
            radius = 2
            if r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w: 
                continue
            histogram = np.zeros(36)
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    angle = 180.0 / np.pi * theta[r + dr, c + dc]
                    if angle < 0: angle = 360 + angle
                    bin_idx = np.clip(np.floor(angle / 10), 0, 35).astype(int)
                    histogram[bin_idx] += magnitude[r+dr,c+dc] / np.sum(magnitude)
            orientation,*_ = np.where(histogram == histogram.max())
            if len(orientation) > 2:
                continue
            for ori in orientation:
                angle = (ori * 10 + 5) * np.pi /180.0
                keypoints.append([r, c, 2, angle])

        # new_img = raw_img.copy()
        # overlay = raw_img.copy()
        # alpha = 0.5

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
                    
                    angle = 180.0 / np.pi * (theta[r + dr_rot, c + dc_rot] - a)
                    if angle < 0: angle = 360 + angle
                    bin_idx = np.clip(np.floor((8.0 / 360) * angle), 0, 7).astype(int)
                    descriptors[i, 32 * int((dc + 8)/4) + 8 * int((dr + 8)/4) + bin_idx] += magnitude[r + dr_rot, c + dc_rot] / np.sum(magnitude)

                    # cv2.circle(overlay, (c + dc_rot, r + dr_rot), 1, (0, 0, 255, 0.1))
                    # size = 2*int(s + 0.5 + 1) + 1
                    # cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(a)), r+int(size * np.sin(a))), (255,0,0))
                if is_out: break
            if is_out: out.append(i)
        keypoints = [keypoints[i] for i in range(len(keypoints)) if i not in out]
        descriptors = [descriptors[i] for i in range(len(descriptors)) if i not in out]
        
        # cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0, new_img)
        # show_image(new_img, 'Histogram - %s' % image_dir)

        descriptors /= np.linalg.norm(descriptors)

        return keypoints, descriptors

    def get_keypoints_log(raw_img):
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        s = 4
        k = 2 ** (1.0 / s)
        sigma_0 = 1.4

        h, w = img.shape
        size_of_octave = s + 3
        height = size_of_octave

        sigma = np.array([sigma_0 * (k ** i) for i in range(size_of_octave)])
        log = np.array([conv.filter(img, conv.get_laplacian_kernel(2*int(sigma[i]) + 1, sigma[i])) for i in range(size_of_octave)])

        # i = 0
        # for l in range(height):
        #     plt.subplot(2,3,i+1)
        #     plt.imshow(log[l])
        #     plt.title('sigma = %.2f' % sigma[l])
        #     plt.xticks([])
        #     plt.yticks([])
        #     i += 1
        # plt.show()

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
                    extrema[l, r, c] = is_extrema(log, l, r, c)

        # new_img = raw_img.copy()
        # for l in range(height):
        #     X, Y = np.where(extrema[l])
        #     for x, y in zip(X, Y):
        #         cv2.circle(new_img, (y, x), 2*int(sigma[l]) + 1, (0, 0, 255))
        # show_image(new_img, 'Extrema - %s' % image_dir)

        Ix = conv.filter(img, conv.sobel_h)
        Iy = conv.filter(img, conv.sobel_v)

        magnitude = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix)

        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy

        # show_image((theta - theta.min()) * 255.0 / (theta.max() - theta.min()), 'theta - %s' % image_dir)

        window_size = 3
        kernel = conv.get_gaussian_kernel(window_size, 1)
        Sxx = conv.filter(Ixx, kernel)
        Sxy = conv.filter(Ixy, kernel)
        Syy = conv.filter(Iyy, kernel)

        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy
        r = det - 0.04*(trace**2)

        thresh_corners = r > .01 * np.max(r)

        keypoints = []
        for l in range(1, height - 1):
            R, C = np.where(np.logical_and(thresh_corners, extrema[l]))
            for r, c in zip(R, C):
                radius = int(np.round(sigma[l]))
                if r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w: 
                    continue
                histogram = np.zeros(36)
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        angle = 180.0 / np.pi * theta[r + dr, c + dc]
                        if angle < 0: angle = 360 + angle
                        bin_idx = np.clip(np.floor(angle / 10), 0, 35).astype(int)
                        histogram[bin_idx] += magnitude[r+dr,c+dc] / np.sum(magnitude)
                orientation,*_ = np.where(histogram == histogram.max())
                if len(orientation) > 2:
                    continue
                for ori in orientation:
                    angle = (ori * 10 + 5) * np.pi /180.0
                    keypoints.append([r, c, sigma[l], angle])

        # new_img = raw_img.copy()
        # overlay = raw_img.copy()
        # alpha = 0.5

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
                    
                    angle = 180.0 / np.pi * (theta[r + dr_rot, c + dc_rot] - a)
                    if angle < 0: angle = 360 + angle
                    bin_idx = np.clip(np.floor((8.0 / 360) * angle), 0, 7).astype(int)
                    descriptors[i, 32 * int((dc + 8)/4) + 8 * int((dr + 8)/4) + bin_idx] += magnitude[r + dr_rot, c + dc_rot] / np.sum(magnitude)

                    # cv2.circle(overlay, (c + dc_rot, r + dr_rot), 1, (0, 0, 255, 0.1))
                    # size = 2*int(s + 0.5 + 1) + 1
                    # cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(a)), r+int(size * np.sin(a))), (255,0,0))
                if is_out: break
            if is_out: out.append(i)
        keypoints = [keypoints[i] for i in range(len(keypoints)) if i not in out]
        descriptors = [descriptors[i] for i in range(len(descriptors)) if i not in out]
        
        # cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0, new_img)
        # show_image(new_img, 'Histogram - %s' % image_dir)

        descriptors /= np.linalg.norm(descriptors)

        return keypoints, descriptors

    def get_keypoints_dog(raw_img):
        img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        s = 4
        k = 2 ** (1.0 / s)
        sigma_0 = 1.4

        h, w = img.shape
        size_of_octave = s + 3
        height = size_of_octave - 1

        sigma = np.array([sigma_0 * (k ** i) for i in range(size_of_octave)])
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

        # new_img = raw_img.copy()
        # for l in range(height):
        #     X, Y = np.where(extrema[l])
        #     for x, y in zip(X, Y):
        #         cv2.circle(new_img, (y, x), 2*int(sigma[l]) + 1, (0, 0, 255))
        # show_image(new_img, 'Extrema - %s' % image_dir)

        Ix = conv.filter(img, conv.sobel_h)
        Iy = conv.filter(img, conv.sobel_v)

        magnitude = np.sqrt(Ix**2 + Iy**2)
        theta = np.arctan2(Iy, Ix)

        Ixx = Ix**2
        Iyy = Iy**2
        Ixy = Ix*Iy

        # show_image((theta - theta.min()) * 255.0 / (theta.max() - theta.min()), 'theta - %s' % image_dir)

        window_size = 3
        kernel = conv.get_gaussian_kernel(window_size, 1)
        Sxx = conv.filter(Ixx, kernel)
        Sxy = conv.filter(Ixy, kernel)
        Syy = conv.filter(Iyy, kernel)

        det = (Sxx * Syy) - (Sxy**2)
        trace = Sxx + Syy
        r = det - 0.04*(trace**2)

        thresh_corners = r > .01 * np.max(r)

        keypoints = []
        for l in range(1, height - 1):
            R, C = np.where(np.logical_and(thresh_corners, extrema[l]))
            for r, c in zip(R, C):
                radius = int(np.round(sigma[l]))
                if r - radius < 0 or r + radius >= h or c - radius < 0 or c + radius >= w: 
                    continue
                histogram = np.zeros(36)
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        angle = 180.0 / np.pi * theta[r + dr, c + dc]
                        if angle < 0: angle = 360 + angle
                        bin_idx = np.clip(np.floor(angle / 10), 0, 35).astype(int)
                        histogram[bin_idx] += magnitude[r+dr,c+dc] / np.sum(magnitude)
                orientation,*_ = np.where(histogram == histogram.max())
                if len(orientation) > 2:
                    continue
                for ori in orientation:
                    angle = (ori * 10 + 5) * np.pi /180.0
                    keypoints.append([r, c, sigma[l], angle])

        # new_img = raw_img.copy()
        # overlay = raw_img.copy()
        # alpha = 0.5

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
                    
                    angle = 180.0 / np.pi * (theta[r + dr_rot, c + dc_rot] - a)
                    if angle < 0: angle = 360 + angle
                    bin_idx = np.clip(np.floor((8.0 / 360) * angle), 0, 7).astype(int)
                    descriptors[i, 32 * int((dc + 8)/4) + 8 * int((dr + 8)/4) + bin_idx] += magnitude[r + dr_rot, c + dc_rot] / np.sum(magnitude)

                    # cv2.circle(overlay, (c + dc_rot, r + dr_rot), 1, (0, 0, 255, 0.1))
                    # size = 2*int(s + 0.5 + 1) + 1
                    # cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(a)), r+int(size * np.sin(a))), (255,0,0))
                if is_out: break
            if is_out: out.append(i)
        keypoints = [keypoints[i] for i in range(len(keypoints)) if i not in out]
        descriptors = [descriptors[i] for i in range(len(descriptors)) if i not in out]
        
        # cv2.addWeighted(overlay, alpha, new_img, 1 - alpha, 0, new_img)
        # show_image(new_img, 'Histogram - %s' % image_dir)

        descriptors /= np.linalg.norm(descriptors)

        return keypoints, descriptors

    if feature_extractor == FT_EXT_DOG:
        keypoints, descriptors = get_keypoints_dog(raw_img)
    elif feature_extractor == FT_EXT_LOG:
        keypoints, descriptors = get_keypoints_log(raw_img)
    elif feature_extractor == FT_EXT_HARRIS:
        keypoints, descriptors = get_keypoints_harris(raw_img)

    # show_image(raw_img, 'Original - %s' % image_dir)

    # new_img = raw_img.copy()
    # for r, c, sigma, angle in keypoints:
    #     cv2.circle(new_img, (c, r), 2*int(sigma + 0.5) + 1, (0, 0, 255))
    # show_image(new_img, 'Thresholded - %s' % image_dir)

    # new_img = raw_img.copy()
    # for r, c, sigma, angle in keypoints:
    #     cv2.circle(new_img, (c, r), 2*int(sigma + 0.5) + 1, (0, 0, 255))

    #     size = 2*int(sigma + 0.5 + 1) + 1
    #     cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(angle)), r+int(size * np.sin(angle))), (255,0,0))
    # show_image(new_img, 'Orientation - %s' % image_dir)

    return keypoints, descriptors

def distance(d1, d2):
    return np.linalg.norm(d1 - d2)

FT_EXT_HARRIS = 0
FT_EXT_DOG = 1
FT_EXT_LOG = 2

def resize_image(img):
    img = img.copy()
    h, w = img.shape[:2]
    if h > 500:
        nw = int(w * 500.0 / h)
        nh = 500
        img = cv2.resize(img, (nw, nh))
    return img

if __name__ == "__main__":
    if len(sys.argv) == 4:
        image_dir_1 = sys.argv[1]
        image_dir_2 = sys.argv[2]

        img1 = cv2.imread(image_dir_1, cv2.IMREAD_COLOR)
        img2 = cv2.imread(image_dir_2, cv2.IMREAD_COLOR)

        img1 = resize_image(img1)
        img2 = resize_image(img2)

        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        feature_extractor = int(sys.argv[3])
        kps_1, descriptors_1 = sift(image_dir_1, img1, feature_extractor)
        kps_2, descriptors_2 = sift(image_dir_2, img2, feature_extractor)

        # create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        # combine 2 images
        vis[:h1, :w1,:3] = img1
        vis[:h2, w1:w1+w2,:3] = img2

        # cv2.imshow('Original', vis)

        res = []
        for idx1, d1 in enumerate(descriptors_1):
            ds = []
            for idx2, d2 in enumerate(descriptors_2):
                ds.append((distance(d1, d2), kps_1[idx1], kps_2[idx2]))
            dist, [r1, c1, _, _], [r2, c2, _, _] = sorted(ds)[0]
            res.append((dist, (r1, c1), (r2, c2)))
        
        res = sorted(res)[:20]
        for dist, (r1, c1), (r2, c2) in res:
            cv2.line(vis, (c1, r1), (c2 + w1, r2), (255, 0, 0))

        cv2.imshow('Match', vis)
        cv2.imwrite('match.png', vis)
    elif len(sys.argv) == 3:
        image_dir = sys.argv[1]
        raw_img = cv2.imread(image_dir, cv2.IMREAD_COLOR)
        raw_img = resize_image(raw_img)
        feature_extractor = int(sys.argv[2])
        keypoints, descriptors = sift(image_dir, raw_img, feature_extractor)

        new_img = raw_img.copy()
        for r, c, sigma, angle in keypoints:
            cv2.circle(new_img, (c, r), 2*int(sigma + 0.5) + 1, (0, 0, 255))

            size = 2*int(sigma + 0.5 + 1) + 1
            cv2.arrowedLine(new_img, (c, r), (c+int(size * np.cos(angle)), r+int(size * np.sin(angle))), (255,0,0))
        show_image(new_img, 'Result')

    cv2.waitKey(0)
    cv2.destroyAllWindows()