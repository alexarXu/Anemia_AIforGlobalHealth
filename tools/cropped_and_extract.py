import os
import cv2
import csv
import numpy as np

'''
This script is used to segment the images with masks and calculate the average RGB values of the segmented images.
'''

def segment_image(original_image_path, mask_image_path, output_image_path):
    '''
    This function segments the original image using the mask image and saves the segmented image.

    Args:
        original_image_path (str): The path to the original image.
        mask_image_path (str): The path to the mask image.
        output_image_path (str): The path to save the segmented image.

    Returns:
        bool: True if successful, False otherwise.
    '''
    original_image = cv2.imread(original_image_path)
    mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

    if original_image is None or mask_image is None:
        print(f"Error reading image or mask for {original_image_path}")
        return False
    
    # Make sure the mask is binary
    _, binary_mask = cv2.threshold(mask_image, 0, 255, cv2.THRESH_BINARY)

    segmented_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
    cv2.imwrite(output_image_path, segmented_image)
    return True


def calculate_average_rgb(image_path):
    '''
    This function calculates the average RGB values of an image.

    Args:
        image_path (str): The path to the image.

    Returns:
        numpy.ndarray: The average RGB values as a NumPy array.
    '''
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error reading image for {image_path}")
        return None

    average_rgb = np.mean(image, axis=(0, 1))
    return average_rgb



folder_name = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/Images_left_palm_corrected/'
mask_folder_name = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/left_palm_segmasks/SegmentationClass/'
output_folder_name = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/Images_left_palm_cropped/'

num = 0
without_mask_path = []
feature_list = []

if not os.path.exists(output_folder_name):
    os.makedirs(output_folder_name)

for image in os.listdir(folder_name):
    original_image_path = os.path.join(folder_name, image)
    mask_image_path = os.path.join(mask_folder_name, image[:-4] + '.png')
    output_image_path = os.path.join(output_folder_name, image)

    if not os.path.exists(mask_image_path):
        print('No mask image for ' + image)
        num += 1
        without_mask_path.append(image)
        continue

    # Segment the image using the mask
    segments = segment_image(original_image_path, mask_image_path, output_image_path)
    if not segments:
        print(f"Failed to segment image: {image}")
        continue

    # Calculate the average RGB values of the segmented image
    average_rgb = calculate_average_rgb(original_image_path)
    if average_rgb is not None:
        feature = [image] + average_rgb.tolist()
        feature_list.append(feature)

print('Total number of images without mask: ' + str(num))
print(without_mask_path)




# save the image wtihtout mask list to a csv file
csv_file_path = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/without_mask_images.csv'
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name']) 
    for image in without_mask_path:
        writer.writerow([image])

# save the feature list to a csv file
feature_csv_file_path = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/image_features.csv'
with open(feature_csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image Name', 'Average left palm R', 'Average left palm G', 'Average left palm B'])
    writer.writerows(feature_list)