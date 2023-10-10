#!/usr/bin/env python
# coding: utf-8

# In[1]:


## EDA

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate a synthetic sample dataset (replace this with your own data)
np.random.seed(42)
n_samples = 1000
data = {
    'Age': np.random.randint(18, 65, size=n_samples),
    'Income': np.random.uniform(20000, 100000, size=n_samples),
    'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_samples),
    'Gender': np.random.choice(['Male', 'Female'], size=n_samples),
    'Country': np.random.choice(['USA', 'Canada', 'UK', 'Australia'], size=n_samples)
}

# Create a DataFrame
df = pd.DataFrame(data)

# Basic Dataset Information
print("First few rows of the dataset:")
print(df.head())

print("\nDataset information:")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

# Missing Data Analysis
print("\nMissing values:")
print(df.isnull().sum())

# Data Visualization
plt.figure(figsize=(12, 6))

# Histograms for numeric columns
plt.subplot(2, 2, 1)
df['Age'].hist(edgecolor='k')
plt.title('Age Distribution')

plt.subplot(2, 2, 2)
df['Income'].hist(edgecolor='k')
plt.title('Income Distribution')

# Box plots for numeric columns
plt.subplot(2, 2, 3)
sns.boxplot(x='Education', y='Income', data=df)
plt.title('Income by Education Level')

# Count plot for categorical columns
plt.subplot(2, 2, 4)
sns.countplot(x='Gender', data=df)
plt.title('Gender Count')

plt.tight_layout()
plt.show()

# Correlation Matrix Heatmap
corr_matrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Categorical Data Analysis
plt.figure(figsize=(8, 6))
sns.countplot(x='Country', data=df)
plt.title('Country Count')
plt.xticks(rotation=45)
plt.show()






