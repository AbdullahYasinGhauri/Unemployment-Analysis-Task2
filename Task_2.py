# Analysis of Unemployment in India

# Importing relevant Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
pd.set_option('display.max_columns', 50)       # Show all columns (up to 50)

# Loading the .csv file
df = pd.read_csv('Unemployment_in_India.csv')
print(df.head())

# Data Cleaning
df.columns = df.columns.str.strip()             # Removing leading & trailing spaces in column names
print(df.columns)
print(df.info)
print('\n\n')

# Data Cleaning
print(f"Duplicates: {df.duplicated().sum()}")   # Checking for duplicate values
print(df.isnull().sum())                        # Checking for null values

df = df.dropna()                                # Dropping rows with missing values
df = df.drop_duplicates()                       # Dropping duplicates values (if any)
df['Date'] = pd.to_datetime(df['Date'])         # Converting "date" to "datetime"
# print(df['Date'].head())                        Checking conversion of "date" to "datetime"
print('\n\n\n')

# Data Exploration & Visualization
print(f"Shape of Unemployment Dataset: {df.shape}")                         # Checking the nums of rows & cols of Dataset
print(f"Unemployment Statistics: {df.describe()}")                          # Checking the Stats of numerical cols

# Plotting a histogram for Unemployment Rate
plt.figure(figsize=(8,5))
sns.histplot(df['Estimated Unemployment Rate (%)'], bins=30, kde=True)
plt.title('Distribution of Unemployment Rate (%)')
plt.xlabel('Unemployment Rate (%)')
plt.show()

# Plotting a barchart of Avg Unemployment Rate against Area
region_avg = df.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values(ascending=False)
plt.figure(figsize=(8,5))
plt.barh(region_avg.index, region_avg.values, color ='skyblue')
plt.title('Average Unemployment Rate (%) by Region')
plt.xlabel('Region')
plt.ylabel('Average Unemployment Rate (%)')
plt.tight_layout()
plt.show()

# Filter Covid period (March 2020 onwards)
covid_period = df[df['Date'] >= '2020-03-01']
pre_covid_period = df[df['Date'] < '2020-03-01']

print(f"Pre-Covid Avg: {pre_covid_period['Estimated Unemployment Rate (%)'].mean():.2f}%")
print(f"Covid Period Avg: {covid_period['Estimated Unemployment Rate (%)'].mean():.2f}%")
print('\n\n')

# Seasonal Trends based on months
df['Month'] = df['Date'].dt.month       # .dt.month lets us take out a part of the date
monthly_avg = df.groupby('Month')['Estimated Unemployment Rate (%)'].mean()
print(monthly_avg)
monthly_avg_df = monthly_avg.reset_index()
plt.figure(figsize=(10,5))
ax = sns.barplot(data= monthly_avg_df, x = 'Month', y = 'Estimated Unemployment Rate (%)', color = 'red')
plt.title('Avg Unemployment Rate By Month')
plt.xlabel('Month')
plt.ylabel('Avg Unemployment Rate (%)')
plt.show()

'''
                                                    INSIGHTS

The overall mean unemployment rate is about 11.79%, with a high standard deviation (10.72) -> shows big variations across
regions & months.

- Covid impact (April & May spike):
The spike to 23.6% in April and 16.6% in May clearly shows the huge impact of the pandemic lockdowns on employment.
After May, rates drop back closer to the ~10% mark. This highlights the need for crisis job programs, or insurance for workers
during pandemics or economic shocks

- Big Regional Differences:
Highest average: Tripura (28.35%), Haryana (26.28%), Jharkhand (20.58%) → these states struggle much more than the national average.
Such big gaps indicate that one-size fits all policies won't work

- Seasonal Trends:
Unemployment jumps in April-May, drop below 10% in other months. This indicates some 
seasonal rural job loss (like agriculture off season).
'''