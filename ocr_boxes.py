import cv2 as cv
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
            crop_name = f"{ocr_images_path}/{str(box_number)}_ocr.jpg"
            cv.imwrite(crop_name, cropped)
            self.show_destroy(f'{crop_name}', cropped)
    