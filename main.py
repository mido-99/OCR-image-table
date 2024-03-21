from box_extractor import BoxExtractor


image_name = r'Image_27445 - 2023 03 17.png'

box_extractor = BoxExtractor(f"image/{image_name}")
box_extractor.run(3)