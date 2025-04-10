import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df=pd.read_csv("datafile.csv") #Read the CSV file

#Performing EDA

print("Data Types and Missing Values:\n")
df.info()
print("\nSummary Statistics:\n",df.describe())
print("\nFirst 5 Rows:\n",df.head())
print("\nLast 5 Rows:\n",df.tail())
print("\nMissing Values in Each Column:\n",df.isnull().sum())
print("\nNumber of Duplicate Rows:",df.duplicated().sum())