import os
import cv2 as cv
import numpy as np
import subprocess
from base import Base


ocr_images_path = r'process_images/ocr_slices'

class OcrBoxes(Base):
    
    def __init__(self, image_path, rects_contours):
        super().__init__(image_path)
        self.rects_contours = rects_contours
    
    def run(self):
        '''Main class function.
        '''
        self.read_image(self.image_path)
        self.convert_contours_to_boxes()
        self.tesseract_work()
        self.export_text()
    
    def show_destroy(self, winName, image):
        '''Show image for debugging
        '''
        cv.imshow(winName, image)
        cv.moveWindow(winName, 300, 500)
        cv.waitKey(0)
        cv.destroyWindow(winName)

    def save_process(self, img, name):
        '''Save given image with name to "process_images/box_extractor/"
        '''
        img_name = f"{ocr_images_path}/{name}.jpg"
        cv.imwrite(img_name, img)
        self.show_destroy(f'{name}', img)
    
    def remove_lines_and_noise(self, image):
        '''
        '''
        pass
    

    def convert_contours_to_boxes(self):
        '''Crop each bounding box of text to a new image for ocr
        '''
        self.boxes = []
        for box_number, contour in enumerate(self.rects_contours):
            x, y, w, h = cv.boundingRect(contour)
            cropped = self.image[y-5:y+h , x-5:x+w]
            crop_name = f"{ocr_images_path}/{str(box_number)}_ocr.jpg"
            inverted = self.process_image(cropped, box_number)
            # noise_removed = self.remove_noise(inverted)        
            # self.show_destroy(f'{crop_name}', inverted)
            cv.imwrite(crop_name, inverted)
    
    def process_image(self, cropped_image, image_number):
        '''Process image before erosion & dilation
        '''
        gray = self.grayscale(cropped_image)
        binary = self.threshold_image(gray)
        inverted = self.invert_image(binary)
        return inverted
    
    def remove_noise(self, image):
        '''
        '''
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        eroded = cv.erode(image, kernel, iterations=1)
        dilated = cv.dilate(eroded, kernel, iterations=2)
        return dilated
    
    def tesseract_work(self):
        '''
        '''
        self.final_text = []
        # ocr_slices = [f'{ocr_images_path}/{i}' for i in os.listdir(ocr_images_path)]
        ocr_slices = [os.path.abspath(f'{ocr_images_path}/{i}') for i in os.listdir(ocr_images_path)]
        tesseract_path = r'"C:\Program Files\Tesseract-OCR\tesseract.exe"'
        for image in ocr_slices:
            #! Error running this command only working command was run from terminal manually was:
# λ "C:\\Program Files\\Tesseract-OCR\\tesseract.exe" "C:\\Users\\Mido Hany\\VS code Projects\\Pyt
# hon\\Full-Projects\\OCR-image-table\\process_images\\ocr_slices\\0_ocr.jpg" out
            result = subprocess.getoutput(f'{tesseract_path} {image} - -l eng').strip()
            self.final_text.append(result)
            
    def export_text(self):
        '''
        '''
        with open('final.txt', 'w') as f:
            for line in self.final_text:
                f.write(line + '\n')
