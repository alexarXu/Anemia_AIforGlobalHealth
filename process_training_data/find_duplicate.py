import pandas as pd
'''
Found there are some duplicate IDs in the dataset. This script is used to find the duplicate IDs.
'''

filepath = 'C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/image_features.csv'
df = pd.read_csv(filepath)

duplicate_values = df['Blood Sample ID'].value_counts()
duplicate_values = duplicate_values[duplicate_values > 1].index
duplicate_rows = df[df['Blood Sample ID'].isin(duplicate_values)]

result_df = duplicate_rows
result_df.to_csv('C:/Users/administer/Desktop/AI_Global/Data/Left_Palm_files/duplicate_IDs.csv', index=False)