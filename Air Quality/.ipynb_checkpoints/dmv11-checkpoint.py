# ---------------------------------------------
# Data Visualization using matplotlib
# Problem: Analyze Air Quality Index (AQI) Trends in a City
# Dataset: City_Air_Quality.csv
# ---------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

# 1. Import dataset
df = pd.read_csv("City_Air_Quality.csv")

# 2. Explore dataset
print(df.head())
print(df.info())

# 3. Identify relevant variables
pollutants = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value", "PM2.5 AQI Value"]

# Handle missing data
df = df.dropna(subset=["City", "AQI Value"])

# ---------------------------------------------
# 4. Visualize overall AQI trend (by City)
# ---------------------------------------------
city_aqi = df.groupby("City")["AQI Value"].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(8,5))
city_aqi.plot(kind="line", marker="o", color="blue")
plt.title("Overall AQI Trend for Top 10 Polluted Cities")
plt.xlabel("City")
plt.ylabel("Average AQI Value")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 5. Plot individual pollutant levels over cities
# ---------------------------------------------
plt.figure(figsize=(10,6))
for pollutant in pollutants:
    plt.plot(df.groupby("City")[pollutant].mean().head(10), label=pollutant)
plt.title("Pollutant Level Trends Across Cities")
plt.xlabel("City")
plt.ylabel("Average Pollutant AQI Value")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 6. Bar plot comparison of AQI across top 10 cities
# ---------------------------------------------
plt.figure(figsize=(8,5))
city_aqi.plot(kind="bar", color="skyblue")
plt.title("AQI Comparison for Top 10 Cities")
plt.xlabel("City")
plt.ylabel("Average AQI Value")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 7. Box plots for pollutant AQI distributions
# ---------------------------------------------
plt.figure(figsize=(8,5))
df.boxplot(column=pollutants)
plt.title("Distribution of Pollutant AQI Values")
plt.ylabel("AQI Value")
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 8. Scatter plot: Relationship between PM2.5 and Overall AQI
# ---------------------------------------------
plt.figure(figsize=(7,5))
plt.scatter(df["PM2.5 AQI Value"], df["AQI Value"], color="orange", alpha=0.5)
plt.title("Relationship between PM2.5 and Overall AQI")
plt.xlabel("PM2.5 AQI Value")
plt.ylabel("Overall AQI Value")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------------------------------
# 9. Customization: Labels, Titles, Legends, Colors (already added)
# ---------------------------------------------
print("\nVisualization complete — all tasks from the problem statement implemented!")
