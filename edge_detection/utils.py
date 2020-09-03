import cv2
import numpy as np

'''
@ Process the image to be displayable
@ img: source image
@ returns displayable image
'''
def transform_visualizable(img):
    # Make a copy of the image
    new_img = img.copy()
    # Clip the part outside [0, 255]
    new_img[new_img > 255] = 255
    new_img[new_img < 0] = 0
    # Return the result with suitable type (uint8)
    return new_img.astype('uint8')

'''
@ Show the image on screen
@ img: source image
@ name_windows: name for the new window
'''
def show_image(img, name_windows):
    # Process the image to be displayable
    new_img = transform_visualizable(img)
    # Show the image on new window with name
    cv2.imshow(name_windows, new_img)