import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import transform_visualizable

'''
@ Helper for the apply, with visualization (before and after)
@ img: source image
@ kernel: the filter to apply
@ visualize: whether to visualize or not
@ return the result image
'''
def convolution(img, kernel, visualize=False):
    filtered_img = img.copy()
    filtered_img = filter(filtered_img, kernel)
    if visualize:
        new_img = transform_visualizable(filtered_img)
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(new_img, cmap='gray')
        plt.show()
        plt.close()
    return filtered_img

'''
@ Apply a filter to an image
@ img: source image
@ kernel: the filter used
@ mode: (reflect, replicate, wrap), mode to fill the image
        padding before applying
@ returns the result image
'''
def filter(img, kernel, mode = 'reflect'):
    # Get image size
    img_h, img_w = img.shape
    
    # Get kernel size
    kernel_h, kernel_w = kernel.shape

    padding_v = kernel_h // 2
    padding_h = kernel_w // 2
    
    # Apply kernel
    filtered = np.zeros((img_h, img_w)).astype('int')
    for r in range(padding_v, img_h - padding_v):
        for c in range(padding_h, img_w - padding_h):
            filtered[r - padding_v][c - padding_h] = np.sum(img[r - padding_v: r + padding_v + 1, c - padding_h: c + padding_h + 1] * kernel)
    return filtered

''' Kernels for convolution '''

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

def Gaussian(x,y,sigma):
    return 1.0 / (2*np.pi*sigma**2) * np.exp(-(x**2 + y**2) * 1.0 / (2 * sigma**2))

def get_gaussian_kernel(size, sigma):
    kernel = np.zeros((size, size))
    radius = size // 2
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            kernel[i + radius, j + radius] = Gaussian(i, j, sigma)
    kernel /= np.sum(kernel)
    return kernel

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

prewitt_h = np.array([[1, 0, -1],
                      [1, 0, -1],
                      [1, 0, -1]])
prewitt_v = np.array([[1, 1, 1],
                      [0, 0, 0],
                      [-1, -1, -1]])  

laplacian_8 = np.array([[-1, -1, -1],
                        [-1, 8, -1],
                        [-1, -1, -1]])

laplacian_4 = np.array([[0, -1, 0],
                        [-1, 4, -1],
                        [0, -1, 0]])    
    