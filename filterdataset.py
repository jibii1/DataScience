import pandas as pd
data=pd.read_csv("MISSING_DATASET_HANDLING.csv",encoding='latin1')
print(data.isnull().sum())