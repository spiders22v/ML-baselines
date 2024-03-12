import os
import pandas as pd
os.chdir("d:/ML-baselines/automl/data")
df = pd.read_csv("classification/bands.csv")

print(df.isnull().head())

print(df.loc[2, "x6"])

print(df.isnull().sum(axis = 0))


print(df.shape)

df = pd.read_csv("classification/bands-1.csv")


print(df.shape)
print(df.isnull().sum(axis = 0))