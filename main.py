import cv2
import numpy as np
import imutils
from solver import *

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from tensorflow.keras.models import load_model

input_size=48

#Read Image
img=cv2.imread('sudoku1.jpg')
cv2.imshow("Input Image: ",img)
cv2.waitKey(0)

def find_board(img):
    """Takes an image as input and finds sudoku board inside it"""
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    bfilter=cv2.bilateralFilter(gray,13,20,20)

    edged= cv2.Canny(bfilter,30,180)
    keypoints= cv2.findContours(edged.copy(), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contours=imutils.grab_contours(keypoints)
    newimg=cv2.drawContours(img.copy(),contours,-1,(0,255,0),3)
    # cv2.imshow("Countour",newimg)

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

board, location = find_board(img)
# cv2.imshow("Board",board)
# cv2.waitKey(0)

#split the board into 81 individual images
print(type(board))
def split_boxes(board):
    """takes  a sudoku board and split it into 81 cells.
    each cell contains an element of that board or an empty cell"""
    rows= np.vsplit(board,9)
    boxes=[]
    for r in rows:
        cols=np.hsplit(r,9)
        for box in cols:
            box=cv2.resize(box, (input_size, input_size))/255.0
            cv2.imshow("Splitted block",box)
            cv2.waitKey(50)

            boxes.append(box)
    return boxes


gray= cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
rois= split_boxes(gray)
rois=np.array(rois).reshape(-1,input_size,input_size,1)

classes = np.arange(0,10)
model=load_model("model-OCR.h5") #OCR prediction model

#get prediction
prediction= model.predict(rois)
# print(prediction)

predicted_numbers=[]
#get classes from prediction
for i in prediction:
    index=(np.argmax(i))
    predicted_number=classes[index]
    predicted_numbers.append(predicted_number)
print(predicted_numbers)


