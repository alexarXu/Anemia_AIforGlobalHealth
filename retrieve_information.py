import os
import cv2
import numpy as np
import pandas as pd
import csv

def read_excel_sheet(file_path, sheet_name):
    """
    读取 Excel 文件中指定名称的 sheet。

    参数:
    file_path (str): Excel 文件路径。
    sheet_name (str): 要读取的 sheet 名称。

    返回:
    DataFrame: 读取的 sheet 数据。
    """
    # 使用 pandas 读取指定 sheet
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    return df

# 示例用法
# excel_file = 'path_to_excel_file.xlsx'
# sheet_name = 'Sheet1'
# df = read_excel_sheet(excel_file, sheet_name)
# print(df)

def get_patient_ids(feature_path, excel_file_path, sheet_name):
    """
    遍历每一个 image name，并在特定 sheet 中找到对应的 patient_id 值。

    参数:
    image_paths (list): 包含图像路径的列表。
    excel_file_path (str): Excel 文件路径。
    sheet_name (str): 要读取的 sheet 名称。

    返回:
    list: 包含每个 image name 对应的 patient_id 值的列表。
    """
    # 读取指定 sheet
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)

    # 获取图像名称列表
    df_feature = pd.read_csv(feature_path)
    image_names = df_feature['Image Name'].tolist()

    # 初始化 patient_id 列表
    patient_ids = []

    # 遍历每一个 image name
    for image_name in image_names:
        # 在 DataFrame 中找到对应的行
        row = df[df['Image_Name'] == image_name]
        if not row.empty:
            # 获取 patient_id 列对应的值
            patient_id = row['Blood_Sample_ID'].values[0]
            patient_ids.append(patient_id)
        else:
            patient_ids.append(None)  # 如果没有找到对应的行，添加 None

    return patient_ids

# 示例用法
# feature_path = 'C:/Users/administer/Desktop/AI_Global/Data/Right_Fingernail_files/image_features.csv'
# excel_file_path = 'C:/Users/administer/Desktop/AI_Global/Data/Anemia_dataset_train.xlsx'
# sheet_name = 'Right_Finger_Nail_Data'

# existing_data = pd.read_csv(feature_path)

# patient_ids = get_patient_ids(feature_path, excel_file_path, sheet_name)
# df_patient_ids = pd.DataFrame(patient_ids, columns=['Blood Sample ID'])
# existing_data = pd.concat([existing_data, df_patient_ids], axis=1)
# existing_data.to_csv(feature_path, index=False)


# def merge_csv_files(csv_paths, output_path, key_column='Blood Sample ID'):
#     """
#     合并多个 CSV 文件，通过指定的列进行匹配。

#     参数:
#     csv_paths (list): 包含 CSV 文件路径的列表。
#     output_path (str): 合并后的 CSV 文件保存路径。
#     key_column (str): 用于匹配的列名，默认为 'Blood Sample ID'。

#     返回:
#     None
#     """
#     # 读取第一个 CSV 文件
#     merged_df = pd.read_csv(csv_paths[0])

#     # 依次读取并合并剩余的 CSV 文件
#     for csv_path in csv_paths[1:]:
#         df = pd.read_csv(csv_path)
#         merged_df = pd.merge(merged_df, df, on=key_column, how='outer')

#     # 保存合并后的 DataFrame 到新的 CSV 文件
#     merged_df.to_csv(output_path, index=False)


def merge_feature_files(file_list, key_column, output_file):
    """
    通过共同的列将多个 CSV 文件合并在一起。

    参数:
    file_list (list): 包含所有 CSV 文件路径的列表。
    key_column (str): 用于合并的共同列名。
    output_file (str): 合并后的 CSV 文件保存路径。

    返回:
    None
    """
    # 初始化一个空的 DataFrame
    merged_df = pd.DataFrame()

    for file in file_list:
        # 读取 CSV 文件
        df = pd.read_csv(file)

        # 删除包含 None 值的行
        df = df.dropna(subset=[key_column])

        # 排除 "Image Name" 列
        df = df.drop(columns=['Image Name'])

        if merged_df.empty:
            merged_df = df
        else:
            # 通过共同的列合并 DataFrame
            merged_df = pd.merge(merged_df, df, on=key_column, how='outer')

    # 保存合并后的 DataFrame 到新的 CSV 文件
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