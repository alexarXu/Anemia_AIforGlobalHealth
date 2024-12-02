import os
import cv2
import numpy as np
import pandas as pd
import csv

'''
This script is ONLY for preprocessing the training data.
'''

def read_excel_sheet(file_path, sheet_name):
    '''
    This function reads a specific sheet from an Excel file.

    Args:
    file_path (str): Path to the Excel file.
    sheet_name (str): Name of the sheet to read.

    Returns:
    pandas.DataFrame: The DataFrame containing the data from the specified sheet
    '''
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df


def get_patient_ids(feature_path, excel_file_path, sheet_name):
    '''
    This function retrieves the patient IDs from an Excel file based on the image names in a CSV file.

    Args:
    feature_path (str): Path to the CSV file containing image features.
    excel_file_path (str): Path to the Excel file containing patient information.
    sheet_name (str): Name of the sheet in the Excel file.

    Returns:
    list: A list of patient IDs corresponding to the image names in the CSV file.
    '''
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    df_feature = pd.read_csv(feature_path)
    image_names = df_feature['Image Name'].tolist()

    patient_ids = []

    for image_name in image_names:
        row = df[df['Image_Name'] == image_name]
        if not row.empty:
            patient_id = row['Blood_Sample_ID'].values[0]
            patient_ids.append(patient_id)
        else:
            patient_ids.append(None)  

    return patient_ids

# need to change the names of the body parts
feature_path = 'C:/Users/administer/Desktop/AI_Global/Data/Right_Fingernail_files/image_features.csv'
excel_file_path = 'C:/Users/administer/Desktop/AI_Global/Data/Anemia_dataset_train.xlsx'
sheet_name = 'Right_Finger_Nail_Data'

existing_data = pd.read_csv(feature_path)

patient_ids = get_patient_ids(feature_path, excel_file_path, sheet_name)
df_patient_ids = pd.DataFrame(patient_ids, columns=['Blood Sample ID'])
existing_data = pd.concat([existing_data, df_patient_ids], axis=1)
existing_data.to_csv(feature_path, index=False)



def merge_feature_files(file_list, key_column, output_file):
    '''
    This function merges multiple CSV files containing image features into a single CSV file. 
    All the data will be merged based on the Patient ID column.

    Args:
    file_list (list): A list of file paths to the CSV files.
    key_column (str): The column name to use for merging the files.
    output_file (str): The path to save the merged CSV file
    '''
    merged_df = pd.DataFrame()

    for file in file_list:
        df = pd.read_csv(file)
        df = df.dropna(subset=[key_column])
        df = df.drop(columns=['Image Name'])

        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=key_column, how='outer')

    merged_df.to_csv(output_file, index=False)




path1 = 'C:/Users/administer/Desktop/AI_Global/Data/Left_eye_files/image_features.csv'
path2 = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/image_features.csv'
path3 = 'C:/Users/administer/Desktop/AI_Global/Data/Right_Fingernail_files/image_features.csv'

csv_files = [path1, path2, path3]
key_column = 'Blood Sample ID'
output_csv = 'C:/Users/administer/Desktop/AI_Global/Data/combined_features.csv'
# merge_feature_files(csv_files, key_column, output_csv)

df_metadata = pd.read_excel('C:/Users/administer/Desktop/AI_Global/Data/Anemia_dataset_train.xlsx', sheet_name='Sheet1')
df_metadata = df_metadata.drop(columns=['index', 'Total Serial Number', 'S No.', 'Unique ID'])
print(df_metadata.head())

df_features = pd.read_csv('C:/Users/administer/Desktop/AI_Global/Data/combined_features.csv')
df_features = df_features.dropna(subset=['Average left eye R', 'Average left palm R', 'Average right nail R'])
print(np.shape(df_features))

df_metadata = pd.merge(df_metadata, df_features, on='Blood Sample ID', how='outer')
df_metadata.to_csv('C:/Users/administer/Desktop/AI_Global/Data/combined_metadata.csv', index=False)

df_metadata = df_metadata.dropna(subset=['Average left eye R'])

df_label = df_metadata[['Blood Sample ID', 'Haemoglobin (in mg/dl)']]
df_label.to_csv('C:/Users/administer/Desktop/AI_Global/Data/label.csv', index=False)
df_data = df_metadata.drop(columns=['Haemoglobin (in mg/dl)'])
df_data['Gender'] = df_data['Gender'].apply(lambda x: 1 if x == 'Male' else 0)
df_data.to_csv('C:/Users/administer/Desktop/AI_Global/Data/data.csv', index=False)