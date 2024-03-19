'''
Demo of Removing table vertical & horizontal lines, in order to get only letters for 
better ocr results
'''
import cv2 as cv
import numpy as np


def show_destroy(winName, img):
    img = cv.resize(img, (750, 700))
    cv.imshow(winName, img)
    cv.moveWindow(winName, 200, 10)
    cv.waitKey(0)
    cv.destroyWindow(winName)

def main():
    src = cv.imread('image/nutrition_table.jpg')
    assert src is not None, "Check file name or path"
    show_destroy('original', src)

    # Grayscale image
    gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
    show_destroy('gray', gray)
    # print(gray)

    #? Binary image 2 ways can work; either a constant threshold or adaptive.
    #? For my case thresh works better due to clear & uniform borders

    #* Method 1
    # cv.threshold
    binary = cv.threshold(gray, 110, 255, cv.THRESH_BINARY)[1]
    inverted = cv.bitwise_not(binary)
    show_destroy('inverted', inverted)

    # Dilate to obtain better objects
    dilated = cv.dilate(inverted, None)
    show_destroy('dilated', dilated)

    #* Method 2
    # cv.adaptiveThreshold
    # gray = cv.bitwise_not(gray)
    # bw = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 13, -9)
    # #show_destroy('inverted' , bw)

    # Save original image, so we work on copies to find horizontal & vertical lines
    horizontal = np.copy(inverted)
    vertical = np.copy(inverted)

    # Create structuring element for horizontal
    horizontal_SE = inverted.shape[1]
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (horizontal_SE//30, 1))

    # Apply morphology operations
    horizontal = cv.erode(horizontal, h_kernel)
    show_destroy('eroded', horizontal)
    horizontal = cv.dilate(horizontal, h_kernel, iterations=3)
    show_destroy('dilated', horizontal)
    #? Erosion must be applied first to get rid of unwanted lines, 
    #? Then dilation is used to fill gaps & make complete lines

    # Create structuring element for vertical
    vertical_SE = inverted.shape[0]
    v_kernel = cv.getStructuringElement(cv.MORPH_RECT, (1, vertical_SE//20))

    # Apply morphology operations
    vertical = cv.erode(vertical, v_kernel)
    show_destroy('eroded', vertical)
    vertical = cv.dilate(vertical, v_kernel)
    show_destroy('dilated', vertical)

    # Combine all lines
    combined = cv.add(horizontal, vertical)
    show_destroy('combined', combined)

    # Thicken lines
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    combined_dilated = cv.dilate(combined, kernel, iterations=5)
    show_destroy('combined dilated', combined_dilated)

    # Remove lines from original image
    without_lines = cv.subtract(inverted, combined_dilated)
    show_destroy('without lines', without_lines)

    # Removing noise & triveal pixels
    noise_removed = cv.erode(without_lines, kernel)
    noise_removed = cv.dilate(noise_removed, kernel, iterations=2)
    show_destroy('noise_removed', noise_removed)

if __name__ == '__main__':
    main()