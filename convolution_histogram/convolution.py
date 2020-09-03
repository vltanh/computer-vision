import cv2
import numpy as np

def filter(img, kernel, mode = 'reflect'):
    # Get kernel size
    kernel_w = len(kernel[0])
    kernel_h = len(kernel)
    if kernel_h % 2 == 0: raise ValueError('kernel height must be an odd number')
    if kernel_w % 2 == 0: raise ValueError('kernel width must be an odd number')

    # Get image size
    img_w = len(img[0])
    img_h = len(img)
    
    # Create padding, default: reflect
    if mode == 'reflect': padding_mode = cv2.BORDER_REFLECT
    elif mode == 'replicate': padding_mode = cv2.BORDER_REPLICATE
    elif mode == 'wrap': padding_mode = cv2.BORDER_WRAP
    else: raise ValueError(mode)

    padding_v = kernel_h // 2
    padding_h = kernel_w // 2
    padded_img = cv2.copyMakeBorder(img, padding_v, padding_v, padding_h, padding_h, padding_mode)

    # Apply kernel
    filtered = img.copy()
    for r in range(padding_v, padding_v + img_h):
        for c in range(padding_h, padding_h + img_w):
            temp = 0
            for i in range(-padding_v, padding_v + 1):
                for j in range(-padding_h, padding_h + 1):
                    temp += padded_img[r + i][c + j] * kernel[i + padding_v][j + padding_h]
            if temp < 0: temp = 256 - temp
            filtered[r - padding_v][c - padding_h] = temp % 256
    return filtered

# Identity
identity = np.array([[0, 0, 0], 
                     [0, 1, 0], 
                     [0, 0, 0]])
# Blur
box_blur = 1/9 * np.ones((3,3))
gaussian_33_blur = 1/16 * np.array([[1, 2, 1],
                                    [2, 4, 2],
                                    [1, 2, 1]])
gaussian_55_blur = 1/256 * np.array([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]])
# Sharpen
sharpen = np.array([[0, -1, 0], 
                    [-1, 5, -1], 
                    [0, -1, 0]])
# Edge detection
sobel_h = np.array([[1, 0, -1],
                    [2, 0, -2],
                    [1, 0, -1]])
sobel_v = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]])       

sobel_h_2 = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])
sobel_v_2 = np.array([[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]])       
    