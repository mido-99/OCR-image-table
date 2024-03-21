import cv2 as cv
import numpy as np


class BoxExtractor:
    
    def __init__(self, image_path):
        self.image_path = image_path
    
    def run(self, start=0, end=None):
        '''Main class function
        '''
        self.read_image(self.image_path)
        self.grayscale(self.image)
        self.save_process(self.gray, '0_gray')
        self.threshold_image(self.gray)
        self.save_process(self.binary, '1_binary')
        self.invert_image(self.binary)
        self.save_process(self.inverted, '2_inverted')
        self.find_horizontal_countours()
        self.save_process(self.with_horizontal_lines, '3_hoizontal')
        self.find_vertical_countours()
        self.save_process(self.with_vertical_lines, '4_vertical')
        self.find_boxes()
        self.save_process(self.combined_dilated, '5_combined')
        self.find_contours()
        self.save_process(self.original_with_all_contourrs, '6_original_with_all_contourrs')
        self.filter_boxes_contours()
        self.draw_rects_on_original(self.image, start, end)
        self.save_process(self.orig_with_rects_only, '7_orig_with_rects_only')

    
    def show_destroy(self, winName, image):
        '''Show image for debugging
        '''
        image = cv.resize(image, (900, 750))
        cv.imshow(winName, image)
        cv.moveWindow(winName, 300, 10)
        cv.waitKey(0)
        cv.destroyWindow(winName)
    
    def save_process(self, img, name):
        '''Save given image with name to "process_images/box_extractor/"
        '''
        img_name = f"process_images/box_extractor/{name}.jpg"
        cv.imwrite(img_name, img)
        self.show_destroy(f'{name}', img)

    def read_image(self, img):
        self.image = cv.imread(img)
        assert self.image is not None, "Check file name or path!"

    def grayscale(self, image):
        '''Gray-scale a BGR image using cvtColor()
        '''
        self.gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    def threshold_image(self, gray):
        '''Convert a grayscale image into a binary using cv.threshold()'''
        self.binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]
    
    def invert_image(self, binary):
        '''Inverte a binary image using cv.bitwise_not() 
        '''
        self.inverted = cv.bitwise_not(binary)
    
    def find_horizontal_countours(self):
        '''Finds horizontal lines of tables in the image
        '''
        horizontal = np.copy(self.inverted)
        horizontal_SE = self.inverted.shape[1]
        h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_SE//30, 1))
        horizontal = cv.erode(horizontal, h_kernel)
        self.with_horizontal_lines = cv.dilate(horizontal, h_kernel, iterations=3)

    def find_vertical_countours(self):
        '''Finds vertical lines of tables in the image
        '''
        vertical = np.copy(self.inverted)
        vertical_SE = self.inverted.shape[0]
        v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_SE//20))
        vertical = cv.erode(vertical, v_kernel)
        self.with_vertical_lines = cv.dilate(vertical, v_kernel)

    def find_boxes(self):
        '''Combine lines making complete, dilated boxes in image
        '''
        combined = cv.add(self.with_horizontal_lines, self.with_vertical_lines)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
        self.combined_dilated = cv.dilate(combined, kernel, iterations=5)
    
    def find_contours(self):
        '''Find & draw contours
        '''
        self.contours, hierachy = cv.findContours(self.combined_dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        self.original_with_all_contourrs = self.image.copy()
        self.original_with_all_contourrs = cv.drawContours(self.original_with_all_contourrs, self.contours, -1, (255, 0, 0), 2)

    def filter_boxes_contours(self):
        '''Approximate -bad- rectangular contours to -good- rects and Exclude small-area rects
        '''
        self.rects = []
        for cont in self.contours:
            peri = cv.arcLength(cont, True)
            approx = cv.approxPolyDP(cont, 0.01*peri, True)
            if len(approx) == 4 and cv.contourArea(cont) > 10* 10**3:
                self.rects.append(cont)

    def draw_rects_on_original(self, image, start: int, end: int):
        '''Show boxes contours on image, to allow choosing start and end rects, 
        mainly for choosing which to send to OCR and ignoring unwanted rects.\n
        (different images shows variable contour results)
        '''
        orig_with_rects_only = image.copy()
        self.orig_with_rects_only = cv.drawContours(orig_with_rects_only, self.rects[start:end], -1, (255, 0, 0), 2) #*


# '''
# #* rects[3:-1]
# we use only some rects in rects[start:end] as different images show different results 
# for depending on image's quality, clarity and resilution.
# So we determine which boxes to start and to end up with, sending them to OCR.
# '''