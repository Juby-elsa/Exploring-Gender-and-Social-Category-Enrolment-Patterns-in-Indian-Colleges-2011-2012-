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

#OBJECTIVE 1

# Grouping SC category data by state and summing male and female values
sc = df.groupby('State')[['Caste-Category - SC - Male', 'Caste-Category - SC - Female']].sum()

# Removing the 'All Districts' entry to keep only valid states
sc = sc.drop('All Districts')

# Plotting SC male vs female student count as a bar plot
sc[['Caste-Category - SC - Male', 'Caste-Category - SC - Female']].plot(
    kind='bar', figsize=(10, 5), color=['teal', 'salmon'])

plt.title('SC Category: Male vs Female')
plt.xlabel('State')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Grouping ST category data by state and summing male and female values
st = df.groupby('State')[['Caste-Category - ST - Male', 'Caste-Category - ST - Female']].sum()
st = st.drop('All Districts')

# Plotting ST male vs female student count
st[['Caste-Category - ST - Male', 'Caste-Category - ST - Female']].plot(
    kind='bar', figsize=(10, 5), color=['teal', 'salmon'])

plt.title('ST Category: Male vs Female ')
plt.xlabel('State')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()

# Grouping OBC category data by state and summing male and female values
obc = df.groupby('State')[['Caste-Category - OBC - Male', 'Caste-Category - OBC - Female']].sum()
obc = obc.drop('All Districts')

# Plotting OBC male vs female student count
obc[['Caste-Category - OBC - Male', 'Caste-Category - OBC - Female']].plot(
    kind='bar', figsize=(10, 5), color=['teal', 'salmon'])

plt.title('OBC Category: Male vs Female')
plt.xlabel('State')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.show()