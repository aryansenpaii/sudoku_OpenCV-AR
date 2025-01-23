import cv2
import numpy as np
import imutils
from solver import *

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model


#Read Image
img=cv2.imread('sudoku1.jpg')
cv2.imshow("Input Image: ",img)

def find_board(img):
    """Takes an image as input and finds sudoku board inside it"""
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bfilter=cv2.bilateralFilter(gray,13,20,20)

    edged= cv2.Canny(bfilter,30,180)
    keypoints= cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=imutils.grab_contours(keypoints)
    newimg=cv2.drawContours(img.copy(),contours,-1,(0,255,0),3)
    cv2.imshow("Countour",newimg)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    # Finds rectangular contour
    for contour in contours:
        approx=cv2.approxPolyDP(contour,15,True)
        if len(approx)==4:
            location= approx
            break
    result=get_perspective(img,location)
    return result,location

def get_perspective(img, location, height=900,width=900):
    """takes an image and location of an rectangular board.
    And return the only selected region with a perspective transformation"""
    pts1=np.float32([location[0], location[3], location[1], location[2]])
    pts2= np.float32([[0,0],[width,0], [0,height],[width,height]])

    #Apply perspective Transform Algorithm
    matrix= cv2.getPerspectiveTransform(pts1,pts2)
    result= cv2.warpPerspective(img, matrix, (width,height))
    return result




