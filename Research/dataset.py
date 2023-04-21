import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set the start and end dates for the dataset
start_date = datetime(2010, 1, 1)
end_date = datetime(2022, 4, 7)

# Generate a sequence of dates between the start and end dates
dates = pd.date_range(start_date, end_date, freq='D')

# Generate a sequence of temperatures using a random walk
num_temps = len(dates)
temperatures = np.zeros(num_temps)
temperatures[0] = 20.0
for i in range(1, num_temps):
    temperatures[i] = temperatures[i-1] + np.random.normal(0, 1) * 2

# Create a dataframe to store the weather data
weather_data = pd.DataFrame({'date': dates, 'temperature': temperatures})

# Save the dataframe to a CSV file
weather_data.to_csv('weather.csv', index=False)
