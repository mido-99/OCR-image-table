import cv2 as cv
import numpy as np
import os


def show_destroy(winName, img):
    img = cv.resize(img, (1000, 900))
    cv.imshow(winName, img)
    cv.moveWindow(winName, 200, 10)
    cv.waitKey(0)
    cv.destroyWindow(winName)

src = 'nutrition_table.jpg'
imgs = [i for i in os.listdir() if i.endswith('.png')]
# src = imgs[2]

# for i in imgs:
src = cv.imread(imgs[2])
assert src is not None, "Check file name or path"
show_destroy('original', src)

# Grayscale image
gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
show_destroy('gray', gray)
# print(gray)

# Binary image NOTE 2 ways can work; either a constant thresh or adaptive.
# For my case thresh works better due to clear & uniform borders

# cv.threshold
thresh = cv.threshold(gray, 110, 255, cv.THRESH_BINARY)[1]
binary = cv.bitwise_not(thresh)
binary = cv.dilate(binary, None, iterations=6)
show_destroy('binary', binary)

# Find & draw contours
contours, hierachy = cv.findContours(binary, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
src_with_cntrs = src.copy()
src_with_cntrs = cv.drawContours(src_with_cntrs, contours, -1, (0, 255, 0), 2)
show_destroy('Contours', src_with_cntrs)

# Approximation
rects = []
for cont in contours:
    peri = cv.arcLength(cont, True)
    approx = cv.approxPolyDP(cont, 0.01*peri, True)
    if len(approx) == 4 and cv.contourArea(cont) > 10* 10**4:
        rects.append(cont)
src_rects_only = src.copy()
src_rects_only = cv.drawContours(src_rects_only, rects, -1, (255, 0, 0), 2)
show_destroy('Rects only', src_rects_only)

#cv.adaptiveThreshold
# gray = cv.bitwise_not(gray)
# bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, -9)
# #show_destroy('inverted' , bw)

# Save original image, so we work on copies to find horizontal & vertical lines
horizontal = np.copy(binary)
vertical = np.copy(binary)

# Create structuring element for horizontal
horizontal_SE = binary.shape[1]
h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_SE//30, 1))

# Apply morphology operations
horizontal = cv.erode(horizontal, h_kernel)
# #show_destroy('eroded', horizontal)
horizontal = cv.dilate(horizontal, h_kernel, iterations=3)
# #show_destroy('dilated', horizontal)
#? Erosion must be applied first to get rid of unwanted lines, 
#? Then dilation is used to fill gaps & make complete lines

# Create structuring element for vertical
vertical_SE = binary.shape[0]
v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_SE//20))

# Apply morphology operations
vertical = cv.erode(vertical, v_kernel)
# #show_destroy('eroded', vertical)
vertical = cv.dilate(vertical, v_kernel)
# #show_destroy('dilated', vertical)

# Combine all lines
combined = cv.add(horizontal, vertical)
#show_destroy('combined', combined)

# Thicken lines
kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
combined_dilated = cv.dilate(combined, kernel, iterations=5)
#show_destroy('combined dilated', combined_dilated)

# Remove lines from original image
without_lines = cv.subtract(binary, combined_dilated)
#show_destroy('without lines', without_lines)

# Removing noise & triveal pixels
noise_removed = cv.erode(without_lines, kernel)
noise_removed = cv.dilate(without_lines, kernel, iterations=2)
#show_destroy('noise_removed', noise_removed)

# Finding word