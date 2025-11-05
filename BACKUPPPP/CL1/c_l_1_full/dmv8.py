# Import libraries
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# 1. Set up API
# ----------------------------
API_KEY = ''
city = 'London'
country = 'GB'

# Base URL for current weather data
url = f"http://api.openweathermap.org/data/2.5/forecast?q={city},{country}&appid={API_KEY}&units=metric"

# ----------------------------
# 2. Retrieve Data from API
# ----------------------------
response = requests.get(url)
if response.status_code == 200:
    data = response.json()
else:
    print("Failed to fetch data:", response.status_code)
    data = None

# ----------------------------
# 3. Extract Relevant Attributes
# ----------------------------
# OpenWeatherMap 5-day forecast returns data every 3 hours under 'list'
weather_list = data['list']

# Create a DataFrame
weather_data = pd.DataFrame([{
    'datetime': item['dt_txt'],
    'temperature': item['main']['temp'],
    'temp_min': item['main']['temp_min'],
    'temp_max': item['main']['temp_max'],
    'humidity': item['main']['humidity'],
    'wind_speed': item['wind']['speed'],
    'weather_main': item['weather'][0]['main'],
    'weather_description': item['weather'][0]['description'],
    'precipitation': item.get('rain', {}).get('3h', 0)  # rain in last 3h, if available
} for item in weather_list])

# ----------------------------
# 4. Data Cleaning / Preprocessing
# ----------------------------
weather_data['datetime'] = pd.to_datetime(weather_data['datetime'])
weather_data.fillna(0, inplace=True)

# ----------------------------
# 5. Data Modeling / Analysis
# ----------------------------
# Average temperature over time
avg_temp = weather_data['temperature'].mean()
max_temp = weather_data['temp_max'].max()
min_temp = weather_data['temp_min'].min()
print(f"Average Temp: {avg_temp:.2f}°C, Max Temp: {max_temp:.2f}°C, Min Temp: {min_temp:.2f}°C")

# Aggregate daily statistics
daily_weather = weather_data.resample('D', on='datetime').agg({
    'temperature': 'mean',
    'temp_min': 'min',
    'temp_max': 'max',
    'humidity': 'mean',
    'wind_speed': 'mean',
    'precipitation': 'sum'
})
print(daily_weather)

# ----------------------------
# 6. Visualizations (All in One Page)
# ----------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'Weather Analysis for {city}', fontsize=16, fontweight='bold')

# --- (1) Line Plot: Temperature Trends ---
axes[0, 0].plot(daily_weather.index, daily_weather['temperature'], marker='o', label='Avg Temp')
axes[0, 0].plot(daily_weather.index, daily_weather['temp_max'], marker='x', label='Max Temp')
axes[0, 0].plot(daily_weather.index, daily_weather['temp_min'], marker='^', label='Min Temp')
axes[0, 0].set_title('Temperature Trends')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Temperature (°C)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# --- (2) Bar Plot: Daily Precipitation ---
axes[0, 1].bar(daily_weather.index, daily_weather['precipitation'], color='skyblue')
axes[0, 1].set_title('Daily Precipitation (mm)')
axes[0, 1].set_xlabel('Date')
axes[0, 1].set_ylabel('Precipitation (mm)')
axes[0, 1].grid(True)

# --- (3) Scatter Plot: Temperature vs Humidity ---
sns.scatterplot(x='temperature', y='humidity', data=weather_data, ax=axes[1, 0], color='tomato')
axes[1, 0].set_title('Temperature vs Humidity')
axes[1, 0].set_xlabel('Temperature (°C)')
axes[1, 0].set_ylabel('Humidity (%)')

# --- (4) Heatmap: Correlation ---
sns.heatmap(weather_data[['temperature','temp_min','temp_max','humidity','wind_speed','precipitation']].corr(),
            annot=True, cmap='coolwarm', ax=axes[1, 1])
axes[1, 1].set_title('Correlation between Weather Attributes')

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to fit title
plt.show()


# ----------------------------
# 7. Optional: Geographical Visualization
# ----------------------------
# If you want to map multiple locations, you can use city coordinates
lat = data['city']['coord']['lat']
lon = data['city']['coord']['lon']
print(f"{city} Coordinates: Latitude {lat}, Longitude {lon}")
