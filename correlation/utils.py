import cv2
import numpy as np

def transform_visualizable(img):
    new_img = img.copy()
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    return new_img.astype('uint8')

def show_image(img, name_windows):
    new_img = transform_visualizable(img)
    cv2.imshow(name_windows, new_img)