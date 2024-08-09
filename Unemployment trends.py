import pandas as pd
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

# Load dataset
df = pd.read_csv('unemployment_data.csv')

# Display the first few rows of the dataframe
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop or fill missing values if necessary
df = df.dropna()  # or df.fillna(method='ffill')

# Convert relevant columns to appropriate data types if needed
# e.g., converting dates to datetime format
df['date'] = pd.to_datetime(df['date'])

print(df.info())

# Unemployment Rate by Age
age_group = df.groupby('age')['unemployment_rate'].mean()
plt.figure(figsize=(10, 6))
age_group.plot(kind='bar')
plt.title('Unemployment Rate by Age')
plt.xlabel('Age Group')
plt.ylabel('Unemployment Rate')
plt.show()

# Unemployment Rate by Gender
gender_group = df.groupby('gender')['unemployment_rate'].mean()
plt.figure(figsize=(8, 5))
gender_group.plot(kind='bar', color=['blue', 'pink'])
plt.title('Unemployment Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Unemployment Rate')
plt.show()

# Unemployment Rate by Education
education_group = df.groupby('education')['unemployment_rate'].mean()
plt.figure(figsize=(12, 7))
education_group.plot(kind='bar')
plt.title('Unemployment Rate by Education Level')
plt.xlabel('Education Level')
plt.ylabel('Unemployment Rate')
plt.show()

# Unemployment Rate by Industry
industry_group = df.groupby('industry')['unemployment_rate'].mean()
plt.figure(figsize=(15, 8))
industry_group.plot(kind='bar')
plt.title('Unemployment Rate by Industry')
plt.xlabel('Industry')
plt.ylabel('Unemployment Rate')
plt.show()