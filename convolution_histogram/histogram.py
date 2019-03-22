import cv2
import numpy as np
from matplotlib import pyplot as plt

def get_histogram(img):
    hist = np.zeros(256)
    for row in img:
        for c in row:
            hist[c] += 1
    return hist

def get_transfer_function(img):
    hist = get_histogram(img) 
    h, w = img.shape
    hist = hist / (h * w)
    cdf = np.array([np.sum(hist[:i+1]) for i in range(len(hist))])
    return np.uint8(255 * cdf)

def equalize_histogram(img):
    transfer_function = get_transfer_function(img)
    h, w = img.shape
    new_img = img.copy()
    for r in range(h):
        for c in range(w):
            new_img[r, c] = transfer_function[img[r, c]]
    return new_img

def get_bin_histogram(full_hist, bin = 16):
    step = 256 / bin
    hist = np.zeros(bin)
    for x in range(256):
        idx = int(x / step)
        hist[idx] += full_hist[x]
    bounds = [(int(step*x), int(np.ceil(step*(x+1)-1))) for x in range(bin)]
    return hist, bounds

def plot_histogram(img, bin = 256):
    hist, bounds = get_bin_histogram(get_histogram(img), bin)

    plt.figure(figsize=(20,8))

    plt.subplot(2, 1, 1)
    plt.imshow(img, cmap='gray')

    plt.subplot(2, 1, 2)
    x = ['{}-{}'.format(m, M) for m, M in bounds]
    plt.bar(x,hist,width=1)
    plt.xticks(rotation = 'vertical')
    if bin > 128: plt.xticks([])
    
    plt.show()
    plt.close()