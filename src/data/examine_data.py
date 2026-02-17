import pandas as pd

df = pd.read_excel(r'datasets/open_datasets/ur3+cobotops/dataset_02052023.xlsx')
print('Shape:', df.shape)
print('Columns:', list(df.columns))
print('\nFirst row:')
print(df.iloc[0])
print('\nData types:')
print(df.dtypes)
