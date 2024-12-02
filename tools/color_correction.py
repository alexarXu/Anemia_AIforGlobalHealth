import cv2
import numpy as np
import os
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

'''
Still trying to change the way of color correction.
Possible ways:
1. Click the five points on the color card to get the color values.
2. Use the Hough Transform to detect the color circle and get the color values.
3. Use the Canny edge detection to detect the color circle and get the color values.

The first way is the most direct and simple way, the codes are as follows.
'''

# clicker
def onclick(event):
    '''
    This function is called when the user clicks on the image. It records the coordinates of the click.
    '''
    ix, iy = int(event.xdata), int(event.ydata)
    print(f"Clicked at ({ix}, {iy})")
    click_points.append((ix, iy))
    if len(click_points) == 5:
        plt.close()

def process_image(image_path):
    '''
    This function allows the user to click on 5 points in the image to specify the colors of the color card.
    Then it will calculate a color correction matrix to correct the colors in the image.

    Args:
        image_path (str): The path to the image file.

    Returns:
        None
    '''
    global click_points
    click_points = []  

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    print("Color card coordinates:", click_points)

    color_samples = {
        "blue": {"position": click_points[0], "standard": [0, 0, 255]},
        "green": {"position": click_points[1], "standard": [0, 255, 0]},
        "red": {"position": click_points[2], "standard": [255, 0, 0]},
        "white": {"position": click_points[3], "standard": [255, 255, 255]},
        "black": {"position": click_points[4], "standard": [0, 0, 0]}
    }

    for color_name, color_info in color_samples.items():
        position = color_info["position"]
        color = color_info["standard"]
        cv2.circle(image, position, 5, color, -1)

    # cv2.imshow('Image with Color Samples', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    actual_colors = []
    standard_colors = []

    for color, data in color_samples.items():
        x, y = data["position"]
        patch = image_rgb[y-5:y+5, x-5:x+5]
        actual_rgb = np.mean(patch, axis=(0, 1))
        actual_colors.append(actual_rgb)
        standard_colors.append(data["standard"])

    actual_colors = np.array(actual_colors)
    standard_colors = np.array(standard_colors)

    reg = LinearRegression()
    reg.fit(actual_colors, standard_colors)
    correction_matrix = reg.coef_
    correction_offset = reg.intercept_

    def apply_correction(img, matrix, offset):
        '''
        Apply color correction to an image.

        Args:
            img (numpy.ndarray): The input image.
            matrix (numpy.ndarray): The correction matrix.
            offset (numpy.ndarray): The correction offset.

        Returns:
            numpy.ndarray: The corrected image.
        '''
        img_flat = img.reshape(-1, 3)
        corrected_flat = np.dot(img_flat, matrix.T) + offset
        corrected_flat = np.clip(corrected_flat, 0, 255)
        return corrected_flat.reshape(img.shape).astype(np.uint8)

    corrected_image_rgb = apply_correction(image_rgb, correction_matrix, correction_offset)
    corrected_image = cv2.cvtColor(corrected_image_rgb, cv2.COLOR_RGB2BGR)

    # save_folder = 'C:/Users/administer/Desktop/AI_Global/Data/Left_eye_files/Images_left_eye_corrected/'
    corrected_image_path = image_path.replace('tongue_image_2', 'Images_tongue_corrected')
    cv2.imwrite(corrected_image_path, corrected_image)
    # cv2.imshow('Corrected Image', corrected_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# folder = 'C:/Users/administer/Desktop/AI_Global/Data/Tongue_files/tongue_image_2/'
# for filename in os.listdir(folder):
#     if filename.endswith('.jpg'):
#         image_path = os.path.join(folder, filename)
#         process_image(image_path)

'''
Input the image path when use the function, output the corrected image with the same name in the folder 'Images_tongue_corrected'
'''
process_image('C:/Users/administer/Desktop/AI_Global/Data/Tongue_files/tongue_image_2/20240308_124239.jpg') # input the image path here
