import cv2
import numpy as np
from skimage import measure
import os

'''
This part is used to segment the largest connected component in an image.
'''
image = cv2.imread('C:/Users/administer/Desktop/AI_Global/Data/20241202032758.png', cv2.IMREAD_GRAYSCALE)

blurred = cv2.GaussianBlur(image, (5, 5), 0)

_, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

labels = measure.label(binary, connectivity=2)

properties = measure.regionprops(labels)
largest_region = max(properties, key=lambda x: x.area)

largest_region_mask = np.zeros_like(binary)
largest_region_mask[labels == largest_region.label] = 255

# cv2.imshow('Largest Connected Component', largest_region_mask)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


'''
This part is used to convert the segmented images to black and white images.
'''
# # need to change the names of the body parts
folder= 'C:/Users/administer/Desktop/AI_Global/Data/Right_Fingernail_files/right_nail_segmask/SegmentationClass'

for root, dirs, files in os.walk(folder):
    for file in files:
        image = cv2.imread(os.path.join(root, file))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, black_and_white_image = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)

        cv2.imwrite(os.path.join(root, file), black_and_white_image)
                    
        # cv2.imshow('Black and White Image', black_and_white_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
