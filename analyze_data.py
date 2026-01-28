import pandas as pd

df = pd.read_csv('data/Combined.csv')
print(f'Rows: {len(df)}')
print(f'Columns: {list(df.columns)}')
print(f'Shape: {df.shape}')
print('\nData types:')
print(df.dtypes)
print('\nMissing values:')
print(df.isnull().sum())
print('\nLabel distribution:')
print(df['status'].value_counts())
print('\nFirst few samples:')
print(df.head())
