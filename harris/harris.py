#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:53:49 2019

@author: honghanh
"""

import cv2
import sys
import os
import numpy as np
import preprocessing as pre


def gaussian(size, sigma):
    size = int(size) // 2
    x, y = np.mgrid[-size: size + 1, -size: size + 1]
    gradient = (1 / (2.0 * np.pi * sigma**2)) * (np.exp(-((x**2 + y**2) / (2.0 * sigma**2))))
    return gradient / np.sum(gradient)

# Step 1: Noise Reduction
def filter_img(img, kernel):
    
    
    image_h, image_w = img.shape
    kernel_h, kernel_w = kernel.shape
    
    img_copy = np.zeros((image_h, image_w))
    
    h_value = kernel_h // 2
    w_value = kernel_w // 2
    
    for i in range(h_value, image_h - h_value):
        for j in range(w_value, image_w - w_value):
            temp = 0
            for k in range(-h_value, h_value + 1):
                for l in range(-w_value, w_value + 1):
                    temp += img[i + k][j + l] * kernel[k + h_value][l + w_value]
            img_copy[i][j] = temp
            
    return img_copy

'''
# Use Harris Algorithm to find corners
# @param1 img: image that is converted
# @parem2 window_size: the size of the sliding window

'''

def harrisCornerDetector(img, k, kernelx, kernely):
    
    img_copy = img.copy()
    img_h, img_w = img_copy.shape
    #offset = window_size/2
    #offset = int(offset)
    #cornerList = []
    
    # Step 1: Derivative Calculation
    
    #Ix = cv2.Sobel(img_copy, cv2.CV_64F, 1, 0)
    #Iy = cv2.Sobel(img_copy, cv2.CV_64F, 0, 1)
    
    Ix = filter_img(img_copy, kernelx)
    Iy = filter_img(img_copy, kernely)
    
    #cv2.imshow('Ix', Ix)
    Ixx = Ix**2
    Iyy = Iy**2
    Ixy = Ix*Iy
        
    #Sxx = cv2.GaussianBlur(Ixx, (5,5), 0)
    #Sxy = cv2.GaussianBlur(Ixy, (5,5), 0)
    #Syy = cv2.GaussianBlur(Iyy, (5,5), 0)
    Sxx = filter_img(Ixx, gaussian(5, 5))
    Sxy = filter_img(Ixy, gaussian(5, 5))
    Syy = filter_img(Iyy, gaussian(5, 5))
    
    #cv2.imshow('Gaussian', Sxx)

    # Step 2: Harris response calculation
    det = (Sxx * Syy) - (Sxy**2)
    trace = Sxx + Syy
    r = det - k*(trace**2)
    
    #Step 3: Find edges & corners using R
    
    thresh = abs(r) > .01 * abs(r).max()
    return thresh, r

def computePoints(thresh, r):
    non_zero_cords = np.nonzero(thresh)

    # Tupled Co-ordinates
    tuple_cords = [(i, j) for i,j in zip(non_zero_cords[0], non_zero_cords[1])] 
    # Values at the these co-ordinates
    values = [abs(r)[i, j] for i, j in tuple_cords]
    # Sort the co-ordinates based on the R values
    sorted_cords = [tuple_cords[i] for i in np.argsort(values)[::-1]]
   
    distance = 10  
    # List to hold the cords after NMS
    nms_cords = []
    nms_cords.append(sorted_cords[0])
    for cord in sorted_cords:
        for nms_cord in nms_cords:
            if abs(cord[0]-nms_cord[0]) < distance and abs(cord[1]-nms_cord[1]) < distance:
                break
        else:
            nms_cords.append(cord)
    return nms_cords

def visualizeCornerImage(img, nms_cords):
    for nms_cord in nms_cords:
        cv2.circle(img, (nms_cord[1], nms_cord[0]), 4, (0, 0, 255), -1) 
    return img
