import cv2 as cv
import numpy as np
import subprocess


ocr_images_path = r'process_images/ocr_slices'
tesseract_path = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'

class OcrBoxes:
    
    def __init__(self, original_image, rects_contours):
        self.rects_contours = rects_contours
        self.original_image = original_image
    
    def run(self):
        self.read_image()
        self.convert_contours_to_boxes()
    
    def show_destroy(self, winName, image):
        '''Show image for debugging
        '''
        cv.imshow(winName, image)
        cv.moveWindow(winName, 300, 500)
        cv.waitKey(0)
        cv.destroyWindow(winName)
        
    def read_image(self):
        self.image = cv.imread(self.original_image)
        assert self.image is not None, "Check file name or path!"

    def convert_contours_to_boxes(self):
        '''Crop each bounding box of text to a new image for ocr
        '''
        self.boxes = []
        for box_number, contour in enumerate(self.rects_contours):
            x, y, w, h = cv.boundingRect(contour)
            cropped = self.image[y-5:y+h , x-5:x+w]
            # cropped = self.remove_noise(cropped)
            #! Need to process iamge first! Construct a base class with base methods (gary, ...)
            crop_name = f"{ocr_images_path}/{str(box_number)}_ocr.jpg"
            cv.imwrite(crop_name, cropped)
            self.show_destroy(f'{crop_name}', cropped)
    
    def remove_noise(self, image):
        '''
        '''
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        image = cv.erode(image, kernel, iterations=1)
        self.show_destroy('eroded', image)
        image = cv.dilate(image, kernel, iterations=2)
        self.show_destroy('dilated', image)
        return image
    
    def save_process(self, img, name):
        '''Save given image with name to "process_images/box_extractor/"
        '''
        img_name = f"{ocr_images_path}/{name}.jpg"
        cv.imwrite(img_name, img)
        self.show_destroy(f'{name}', img)
