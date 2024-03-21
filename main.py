from box_extractor import BoxExtractor
from ocr_boxes import OcrBoxes

image_path = 'image/Image_27445 - 2023 03 17.png'

box_extractor = BoxExtractor(image_path )
rects = box_extractor.run(3)

ocr_boxes = OcrBoxes(image_path, rects)
ocr_boxes.run()