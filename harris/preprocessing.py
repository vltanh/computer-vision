#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 18:26:21 2019

@author: honghanh
"""

import cv2
'''
# Convert initial image to gray scale
@input: initial image
#output: converted image 
'''
def convertGrayScale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    return img


