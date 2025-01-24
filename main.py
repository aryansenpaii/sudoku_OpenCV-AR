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
"""there will be 81 predictions in total with each containing array of size 10 
containing predictions for 0-9 digits, the prediction that has higher value will be the 
digit we predicted from rois(region of interest) i.e  the model suggests that that character
is highly probable from the image we fed it!"""
prediction= model.predict(rois)
# print(prediction)

predicted_numbers=[]
#get classes from prediction
for i in prediction:
    index=(np.argmax(i))
    predicted_number=classes[index]
    predicted_numbers.append(predicted_number)
# print(predicted_numbers)

#reshape the list
board_num= np.array(predicted_numbers).astype('uint8').reshape(9,9)
# print(board_num)




def displayNumbers(img, numbers, color=(0,255,0)):
    """Text Content:
        str(numbers[(j * 9) + i]): Converts the number to a string for display.
       Position:
            i * W + int(W / 2): Centers the text horizontally within the cell.
            - int(W / 4): Adjusts the text slightly to the left for better alignment.
            int((j + 0.7) * H): Positions the text vertically within the cell, slightly below the middle.
       Font and Appearance:
            cv2.FONT_HERSHEY_COMPLEX: Specifies the font style.
            2: Font scale (size of the text).
            color: Text color.
            2: Thickness of the text.
            cv2.LINE_AA: Anti-aliasing for smoother text edges."""
    W=int(img.shape[1]/9)
    H= int(img.shape[0]/9)
    for i in range(9):
        for j in range(9):
            if(numbers[(j*9)+i] != 0):
                cv2.putText(img,str(numbers[(j*9)+i]), (i*W+int(W/2)-int((W/4)),int((j+0.7)*H)), cv2.FONT_HERSHEY_COMPLEX,2 , color,2,cv2.LINE_AA)
    return img

def get_InvPerspective(img, masked_num, location, height=900, width=900):
    """takes original image as input"""
    pts1=np.float32([[0,0], [width,0],[0,height],[width, height]])
    pts2= np.float32([location[0], location[3], location[1], location[2]])

    #Apply Perspective Transform Algorithm
    matrix=cv2.getPerspectiveTransform(pts1,pts2)
    #put the transformed image back into the original image
    result= cv2.warpPerspective(masked_num,matrix,(img.shape[1],img.shape[0]))
    return result

try:
    solved_board_nums = get_board(board_num)
    binArr= np.where(np.array(predicted_numbers)>0,0,1)
    print(binArr)
    #get only solved numbers for the solved board
    flat_solved_board_nums=solved_board_nums.flatten()*binArr
    #create a mask
    mask=np.zeros_like(board)
    solved_board_mask = displayNumbers(mask, flat_solved_board_nums)
    cv2.imshow("Solved mask", solved_board_mask)
    # Get Inverse Perspective
    inv = get_InvPerspective(img, solved_board_mask, location)
    # cv2.imshow("mask with original image perspective",inv)


except:
    print("Solution doesn't exist. Model misread the digits")

