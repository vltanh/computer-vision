#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:11:32 2019

@author: honghanh
"""
import cv2
import numpy as np
import sys
import harris

if __name__ == "__main__":
    img_dir = sys.argv[1]
    k = sys.argv[2]

    img = cv2.imread(img_dir)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    

    kernelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.int)
    kernely = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype = np.int)
    #kernely = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype = np.int)

    result, r = harris.harrisCornerDetector(gray_img, float(k), kernelx, kernely)
    points = harris.computePoints(result, r)
    
    final = harris.visualizeCornerImage(img, points)
    
    cv2.imshow('result', final)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
