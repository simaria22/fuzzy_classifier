from os import read
import pandas as pd

file_path = 'C:/Users/mariasiouzou/Documents/ΣΙΟΥΖΟΥ ΜΑΡΙΑ ΠΤΥΧΙΑΚΗ/PyTSK-master/mrastats4DayFeatures-zscore.csv'
df=pd.read_csv(file_path)


df.iloc[:, -1] = df.iloc[:, -1].str.replace('C', '')

# Save the last column values in a numpy array
last_column_array = df.iloc[:, -1].to_numpy()

# Save the rest of the DataFrame in another numpy array
rest_of_df_array = df.iloc[:, :-1].to_numpy()

last_column_array=last_column_array.reshape(-1,1)
print(rest_of_df_array.shape)
print(last_column_array)
