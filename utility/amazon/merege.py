import pandas as pd

# Read in the two CSV files
df1 = pd.read_csv('../../data/am1-train.csv')
df2 = pd.read_csv('../../data/home-train.csv')

# Concatenate the two dataframes
result = pd.concat([df1, df2])

# Write the concatenated data to a new CSV file
result.to_csv('../../data/train.csv', index=False)
