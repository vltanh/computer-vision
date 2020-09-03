import cv2
import os
import convolution as conv
import histogram as hist
from matplotlib import pyplot as plt
import numpy as np

def conv_demo(img_dir, kernel, visualize=True):
    # Read image
    img = cv2.imread(img_dir, 0)

    # Apply kernel
    new_img = conv.filter(img, kernel)

    # Show image after change
    if visualize:
        plt.figure(figsize=(16, 8))
        plt.subplot(1, 2, 1)
        plt.imshow(img, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(new_img, cmap='gray')
        plt.show()
        plt.close()

    return new_img

def conv_demo_2(img_dir):
    # Read image
    img = cv2.imread(img_dir, 0)

    plt.figure(figsize=(16, 8))

    i = 0
    for kernel in [conv.box_blur, conv.gaussian_33_blur, conv.gaussian_55_blur, conv.sharpen, conv.sobel_h, conv.sobel_v]:
        # Apply kernel
        new_img = conv.filter(img, kernel)

        # Show image after change
        plt.subplot(2, 3, i + 1)
        plt.imshow(new_img, cmap='gray')

        i += 1
    plt.show()
    plt.close()

    return new_img

def hist_demo(img_dir, visualize=True):
    # Read image
    img = cv2.imread(img_dir, 0)
    h = hist.get_histogram(img)

    # Equalize histogram
    new_img = hist.equalize_histogram(img)
    new_h = hist.get_histogram(new_img)

    # Visualize
    if visualize:
        plt.figure(figsize=(16,8))

        plt.subplot(2, 2, 1)
        plt.imshow(img, cmap='gray')

        plt.subplot(2, 2, 2)
        plt.imshow(new_img, cmap='gray')

        plt.subplot(2, 2, 3)
        plt.bar([x for x in range(256)], h, width=1)
        plt.xticks(rotation = 'vertical')
        
        plt.subplot(2, 2, 4)
        plt.bar([x for x in range(256)], new_h, width=1)
        plt.xticks(rotation = 'vertical')
        
        plt.show()
        plt.close()

    return new_img

def norm(x):
    m = np.min(x)
    M = np.max(x)
    return np.uint8(256 * (x - m) / (M - m))

if __name__ == "__main__":
    img_dir = './input/SanFrancisco.jpg'

    img = cv2.imread(img_dir, 0)
    # dx = conv.filter(img, conv.gaussian_55_blur)
    # dx = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    dx = conv_demo(img_dir, conv.sobel_h, visualize=False)
    dy = conv_demo(img_dir, conv.sobel_v, visualize=False)
    
    cv2.imshow('dx', dx)
    cv2.imshow('dy', dy)

    d = np.uint8(np.sqrt(dx**2 + dy**2) + 0.5)
    cv2.imshow('d', d)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
if __name__ != "__main__":
    img_dir = input('Image directory? ')
    print('--------------------------')
    print('Choose demo: ')
    print('\t(1) Convolution')
    print('\t(2) Histogram')
    choice = 0
    while choice not in [1, 2]: 
        choice = int(input('Your choice (1-2): '))
    print('--------------------------')
    if choice == 1:
        print('Choose kernel: ')
        print('\t(1) Box blur')
        print('\t(2) Gaussian 3x3 blur')
        print('\t(3) Gaussian 5x5 blur')
        print('\t(4) Sharpen')
        print('\t(5) Sobel horizontal edge detection')
        print('\t(6) Sobel vertical edge detection')
        kernels = [conv.box_blur, conv.gaussian_33_blur, conv.gaussian_55_blur, conv.sharpen, conv.sobel_h, conv.sobel_v]
        choice = 0
        while choice not in [i for i in range(1, 7)]: 
            choice = int(input('Your choice (1-6): '))
        print('--------------------------')
        print('Working...')
        new_img = conv_demo(img_dir, kernels[choice - 1])
        print('Done!')
    elif choice == 2:
        print('Working...')
        new_img = conv_demo_2(img_dir)
        print('Done!')

    print('--------------------------')
    choice = input('Do you want to save the new image (Y/N)? ')
    if choice == 'Y':
        name, ext = os.path.splitext(img_dir)
        cv2.imwrite('{}_new{}'.format(name, ext), new_img)