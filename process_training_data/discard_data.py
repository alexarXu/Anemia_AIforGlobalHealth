import os
import numpy as np
import pandas as pd
import cv2

'''
This script is ONLY for preprocessing the training data.
'''

def get_files_in_folder(folder_path):
    '''
    This function returns a dictionary containing the file names in a folder (without the file extension).
    '''
    files = os.listdir(folder_path)
    file_map = {os.path.splitext(file)[0]: file for file in files}
    return file_map

def delete_files(folder_path, files_to_delete, file_map):
    '''
    Tbhis function deletes files from a folder based on a list of file names.

    Args:
    folder_path (str): Path to the folder containing the files.
    files_to_delete (list): List of file names to delete.
    file_map (dict): Dictionary containing the file names in the folder.
    '''
    for file_name in files_to_delete:
        full_file_name = file_map[file_name]
        file_path = os.path.join(folder_path, full_file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {full_file_name}")

def main(folder1, folder2):
    '''
    This function compares two folders and deletes files that are not present in both folders.

    Args:
    folder1 (str): Path to the first folder.
    folder2 (str): Path to the second folder.
    '''
    file_map1 = get_files_in_folder(folder1)
    file_map2 = get_files_in_folder(folder2)

    files_in_folder1 = set(file_map1.keys())
    files_in_folder2 = set(file_map2.keys())

    common_files = files_in_folder1.intersection(files_in_folder2)

    files_to_delete_in_folder1 = files_in_folder1 - common_files
    files_to_delete_in_folder2 = files_in_folder2 - common_files

    delete_files(folder1, files_to_delete_in_folder1, file_map1)
    delete_files(folder2, files_to_delete_in_folder2, file_map2)


if __name__ == "__main__":
    folder1 = "C:/Users/administer/Desktop/AI_Global/Data/Right_Fingernail_files/Images_right_fingernail_corrected"
    folder2 = "C:/Users/administer/Desktop/AI_Global/Data/Right_Fingernail_files/right_nail_segmask/SegmentationClass"  
    main(folder1, folder2)
