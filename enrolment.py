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

#OBJECTIVE 1--Gender-wise Comparison of Caste Categories

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

#Objective 2--Visualizing Minority Group Distribution

# Calculating total male and female counts for each minority group
muslim_male = df['Out of Total - Muslim - Male'].sum()
muslim_female = df['Out of Total - Muslim - Female'].sum()

minority_male = df['Out of Total - Other Minority Communities - Male'].sum()
minority_female = df['Out of Total - Other Minority Communities - Female'].sum()

foreign_male = df['Out of Total - Foreign Students - Male'].sum()
foreign_female = df['Out of Total - Foreign Students - Female'].sum()

# Creating a figure for the 3 pie charts
plt.figure(figsize=(12, 6))
plt.suptitle("Gender-wise Distribution Among Muslim, Minority, and Foreign Students", fontsize=15)

# Pie chart for Muslim students
plt.subplot(1, 3, 1)
plt.pie([muslim_male, muslim_female], labels=['Male', 'Female'], autopct='%1.1f%%', colors=['royalblue', 'hotpink'])
plt.title('Muslim Students')

# Pie chart for Other Minority students
plt.subplot(1, 3, 2)
plt.pie([minority_male, minority_female], labels=['Male', 'Female'], autopct='%1.1f%%', colors=['steelblue', 'lightcoral'])
plt.title('Other Minority Students')

# Pie chart for Foreign students
plt.subplot(1, 3, 3)
plt.pie([foreign_male, foreign_female], labels=['Male', 'Female'], autopct='%1.1f%%', colors=['dodgerblue', 'plum'])
plt.title('Foreign Students')

# Adjusting the  layout to prevent overlapping
plt.tight_layout()
plt.show()

#Objective 3--Heatmap to Show Correlation Between Categories
# Grouping data by district and calculate total values for each caste category
grouped_heatmap = df.groupby('District')[[
    'Caste-Category - SC - Male', 'Caste-Category - SC - Female',
    'Caste-Category - ST - Male', 'Caste-Category - ST - Female',
    'Caste-Category - OBC - Male', 'Caste-Category - OBC - Female',
    'Caste-Category - Total - Male', 'Caste-Category - Total - Female'
]].sum()

# Calculating percentage distribution within each gender group
grouped_heatmap['SC Male %'] = grouped_heatmap['Caste-Category - SC - Male'] / grouped_heatmap['Caste-Category - Total - Male']
grouped_heatmap['SC Female %'] = grouped_heatmap['Caste-Category - SC - Female'] / grouped_heatmap['Caste-Category - Total - Female']
grouped_heatmap['ST Male %'] = grouped_heatmap['Caste-Category - ST - Male'] / grouped_heatmap['Caste-Category - Total - Male']
grouped_heatmap['ST Female %'] = grouped_heatmap['Caste-Category - ST - Female'] / grouped_heatmap['Caste-Category - Total - Female']
grouped_heatmap['OBC Male %'] = grouped_heatmap['Caste-Category - OBC - Male'] / grouped_heatmap['Caste-Category - Total - Male']
grouped_heatmap['OBC Female %'] = grouped_heatmap['Caste-Category - OBC - Female'] / grouped_heatmap['Caste-Category - Total - Female']

# Selecting only percentage columns for correlation
percentage_data = grouped_heatmap[[col for col in grouped_heatmap.columns if '%' in col]]

# Computing the correlation matrix
corr = percentage_data.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="Blues", linewidths=0.5)
plt.title('Correlation Between Caste Categories (Normalized by Gender)')
plt.tight_layout()
plt.show()
