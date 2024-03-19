import cv2 as cv
import numpy as np
import os


def show_destroy(winName, img):
    img = cv.resize(img, (900, 750))
    cv.imshow(winName, img)
    cv.moveWindow(winName, 300, 10)
    cv.waitKey(0)
    cv.destroyWindow(winName)

def invert_img(img):
    '''Inverte a binary image using cv.bitwise_not() After applying gray-scaling & 
    a constant threshold value'''

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    binary = cv.threshold(gray, 120, 255, cv.THRESH_BINARY)[1]
    inverted = cv.bitwise_not(binary)
    return inverted

imgs = [i for i in os.listdir('image') if i.endswith('.png')]
imgs = [os.path.join(os.path.abspath('image'), img) for img in imgs]

src = cv.imread(imgs[2])
assert src is not None, "Check file name or path"
show_destroy('original', src)

inverted = invert_img(src)
dilated = cv.dilate(inverted, None, iterations=6)
show_destroy('Dilated', dilated)

# Find & draw contours
contours, hierachy = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
src_with_cntrs = src.copy()
src_with_cntrs = cv.drawContours(src_with_cntrs, contours, -1, (255, 0, 0), 2)
show_destroy('Contours', src_with_cntrs)

# # Approximation
# rects = []
# for cont in contours:
#     peri = cv.arcLength(cont, True)
#     approx = cv.approxPolyDP(cont, 0.01*peri, True)
#     if len(approx) == 4 and cv.contourArea(cont) > 10* 10**4:
#         rects.append(cont)
# src_rects_only = src.copy()
# src_rects_only = cv.drawContours(src_rects_only, rects, -1, (255, 0, 0), 2)
# show_destroy('Rects only', src_rects_only)

