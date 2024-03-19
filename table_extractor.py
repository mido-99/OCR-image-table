import cv2 as cv
import numpy as np
import os


def show_destroy(winName, img):
    '''Show image for debugging
    '''
    img = cv.resize(img, (900, 750))
    cv.imshow(winName, img)
    cv.moveWindow(winName, 300, 10)
    cv.waitKey(0)
    cv.destroyWindow(winName)

def invert_img(img):
    '''Inverte a binary image using cv.bitwise_not() After applying gray-scaling & 
    a constant threshold value
    '''
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]
    inverted = cv.bitwise_not(binary)
    return inverted

# Get list of image names in folder so as to easily iterate over them by changing index
imgs = [i for i in os.listdir('image') if i.endswith('.png')]
imgs = [os.path.join(os.path.abspath('image'), img) for img in imgs]
src = cv.imread(imgs[0])
assert src is not None, "Check file name or path"
# show_destroy('original', src)

inverted = invert_img(src)
# show_destroy('inverted', inverted)

horizontal = np.copy(inverted)
horizontal_SE = inverted.shape[1]
h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_SE//30, 1))
horizontal = cv.erode(horizontal, h_kernel)
# show_destroy('eroded', horizontal)
horizontal = cv.dilate(horizontal, h_kernel, iterations=3)
# show_destroy('dilated', horizontal)

vertical = np.copy(inverted)
vertical_SE = inverted.shape[0]
v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_SE//20))
vertical = cv.erode(vertical, v_kernel)
# show_destroy('eroded', vertical)
vertical = cv.dilate(vertical, v_kernel)
# show_destroy('dilated', vertical)

combined = cv.add(horizontal, vertical)
# show_destroy('combined', combined)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
combined_dilated = cv.dilate(combined, kernel, iterations=5)
show_destroy('combined dilated', combined_dilated)

# Find & draw contours
contours, hierachy = cv.findContours(combined_dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
src_with_cntrs = src.copy()
src_with_cntrs = cv.drawContours(src_with_cntrs, contours, -1, (255, 0, 0), 2)
show_destroy('Contours', src_with_cntrs)

# Approximation
rects = []
for cont in contours:
    peri = cv.arcLength(cont, True)
    approx = cv.approxPolyDP(cont, 0.01*peri, True)
    if len(approx) == 4 and cv.contourArea(cont) > 10* 10**3:
        rects.append(cont)

def show_contours(img, contours, start: int, end: int):
    '''Show rectangular contours on image, to allow which is our start and end rects, 
    mainly for choosing which to send to OCR.\n
    (different images shows variable contour results)
    '''
    src_rects_only = img.copy()
    src_rects_only = cv.drawContours(src_rects_only, contours[start:end], -1, (255, 0, 0), 2) #*
    show_destroy('Rects only', src_rects_only)
show_contours(src, rects, 3, -1)

'''
#* rects[3:-1]
we use only some rects in rects[start:end] as different images show different results 
for depending on image's quality, clarity and resilution.
So we determine which boxes to start and to end up with, sending them to OCR.
'''