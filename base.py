import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


class Base(ABC):
    '''Parent class for implementing default methods
    '''
    def __init__(self, image_path):
        self.image_path = image_path
    
    @abstractmethod
    def show_destroy(self, winName, image):
        '''Show image for debugging
        '''
        pass

    @abstractmethod
    def save_process(self, img, name):
        '''Save given image with name to "process_images/box_extractor/"
        '''
        pass
    
    @abstractmethod
    def run(self):
        '''Main class function.
        '''
        pass
    
    def read_image(self, img):
        '''Expect an image path to be read using cv.imread()
        '''
        self.image = cv.imread(img)
        assert self.image is not None, "Check file name or path!"
    
    def grayscale(self, original_image):
        '''Gray-scale a BGR image using cvtColor()
        '''
        return cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)
    
    def threshold_image(self, gray_image):
        '''Convert a grayscale image into a binary using cv.threshold()
        '''
        return cv.threshold(gray_image, 120, 255, cv.THRESH_BINARY)[1]
    
    def invert_image(self, binary_image):
        '''Inverte a binary image using cv.bitwise_not() 
        '''
        return cv.bitwise_not(binary_image)
    
