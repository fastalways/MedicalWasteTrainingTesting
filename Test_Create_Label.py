import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from os import listdir
from os.path import isfile, join
import copy

from numpy.lib.shape_base import split

class cvRect:
    def __init__(self, xywh):
        self.x = xywh[0]
        self.y = xywh[1]
        self.w = xywh[2]
        self.h = xywh[3]
        self.xmin = self.x
        self.ymin = self.y
        self.xmax = self.x + self.w
        self.ymax = self.y + self.h
    def area(self):
        return self.w * self.h
    def tl(self):
        return [self.x,self.y]
    def br(self):
        return [self.x+self.w,self.y+self.h]
    def center(self):
        return [self.x+(self.w//2),self.y+(self.h//2)]
    def get_xywh(self):
        return  [self.x,self.y,self.w,self.h]

dictLabel = {
    'testclass1':0,
    'testclass2':1,
    'testclass3':2,
    'testclass4':3,
}
def makeLabelYOLO(xywh,nameClass,IMAGE_SIZE):
    ''' makeLabelYOLO(xywh<-cvRect,nameClass<-string,IMAGE_SIZE<-numpy.shape())
    '''
    # Yolo Format
    ''' Label_ID X_CENTER_NORM Y_CENTER_NORM WIDTH_NORM HEIGHT_NORM Label_ID_2 X_CENTER_NORM
    X_CENTER_NORM = X_CENTER_ABS/IMAGE_WIDTH 
    Y_CENTER_NORM = Y_CENTER_ABS/IMAGE_HEIGHT 
    WIDTH_NORM = WIDTH_OF_LABEL_ABS/IMAGE_WIDTH 
    HEIGHT_NORM = HEIGHT_OF_LABEL_ABS/IMAGE_HEIGHT
    '''
    global cvRect,dictLabel
    IMAGE_WIDTH = IMAGE_SIZE[1]
    IMAGE_HEIGHT = IMAGE_SIZE[0]
    # find Label_ID
    Label_ID = dictLabel[nameClass]
    X_CENTER_NORM = xywh.center()[0]/IMAGE_WIDTH 
    Y_CENTER_NORM = xywh.center()[1]/IMAGE_HEIGHT 
    WIDTH_NORM = xywh.w/IMAGE_WIDTH 
    HEIGHT_NORM = xywh.h/IMAGE_HEIGHT
    return "%d %.6f %.6f %.6f %.6f" % (Label_ID,X_CENTER_NORM,Y_CENTER_NORM,WIDTH_NORM,HEIGHT_NORM)
def main():
    global cvRect
    TestRect = cvRect([182,66,296-182,159-66])
    print(makeLabelYOLO(TestRect,'testclass4',[273,312]))

if __name__=="__main__":
    main()