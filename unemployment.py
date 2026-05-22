# =========================================================
# UNEMPLOYMENT ANALYSIS WITH PYTHON
# =========================================================

# -----------------------------
# IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("Unemployment in India.csv")

# -----------------------------
# DISPLAY FIRST 5 ROWS
# -----------------------------
print("First 5 Rows of Dataset:\n")
print(df.head())

# -----------------------------
# DATASET INFORMATION
# -----------------------------
print("\nDataset Information:\n")
print(df.info())

# -----------------------------
# CHECK MISSING VALUES
# -----------------------------
print("\nMissing Values:\n")
print(df.isnull().sum())

# -----------------------------
# DATA CLEANING
# -----------------------------

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Drop missing values
df = df.dropna()

# Convert Date column into datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Display cleaned dataset info
print("\nCleaned Dataset:\n")
print(df.info())

print("\nStatistical Summary:\n")
print(df.describe())

plt.figure(figsize=(10,6))

sns.histplot(
    df['Estimated Unemployment Rate (%)'],
    kde=True,
    bins=20
)

plt.title("Distribution of Unemployment Rate")
plt.xlabel("Unemployment Rate (%)")
plt.ylabel("Frequency")

plt.show()

state_avg = df.groupby('Region')[
    'Estimated Unemployment Rate (%)'
].mean().sort_values()

plt.figure(figsize=(14,7))

state_avg.plot(kind='bar', color='skyblue')

plt.title("Average Unemployment Rate by Region")
plt.xlabel("Region")
plt.ylabel("Average Unemployment Rate (%)")

plt.xticks(rotation=90)

plt.show()

# Filter data from March 2020 onwards
covid_data = df[df['Date'] >= '2020-03-01']

plt.figure(figsize=(14,7))

sns.lineplot(
    data=covid_data,
    x='Date',
    y='Estimated Unemployment Rate (%)'
)

plt.title("Impact of COVID-19 on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")

plt.show()

# Extract month from Date
df['Month'] = df['Date'].dt.month

monthly_avg = df.groupby('Month')[
    'Estimated Unemployment Rate (%)'
].mean()

plt.figure(figsize=(10,6))

monthly_avg.plot(
    marker='o',
    linestyle='-'
)

plt.title("Monthly Average Unemployment Trend")
plt.xlabel("Month")
plt.ylabel("Average Unemployment Rate (%)")

plt.grid(True)

plt.show()

# Heatmap of Unemployment by Region and Month

pivot_table = df.pivot_table(
    values='Estimated Unemployment Rate (%)',
    index='Region',
    columns='Month'
)

plt.figure(figsize=(14,8))

sns.heatmap(
    pivot_table,
    cmap='coolwarm',
    annot=False
)

plt.title("Region-wise Monthly Unemployment Heatmap")

plt.show()

#Top 10 Regions with Highest Unemployment

top_regions = state_avg.sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))

top_regions.plot(kind='bar', color='red')

plt.title("Top 10 Regions with Highest Unemployment")
plt.xlabel("Region")
plt.ylabel("Unemployment Rate (%)")

plt.xticks(rotation=45)

plt.show()

# Employment Trend Over Time

plt.figure(figsize=(14,7))

sns.lineplot(
    data=df,
    x='Date',
    y='Estimated Employed'
)

plt.title("Estimated Employment Over Time")
plt.xlabel("Date")
plt.ylabel("Estimated Employed")

plt.show()

print("\n================ INSIGHTS ================\n")
print("1. COVID-19 caused a sharp increase in unemployment rates.")
print("2. Urban and industrial regions showed higher unemployment spikes.")
print("3. Some regions maintained stable employment due to agriculture and local industries.")
print("4. Monthly trends indicate seasonal fluctuations in unemployment.")
print("5. Employment levels gradually recovered after lockdown periods.")
