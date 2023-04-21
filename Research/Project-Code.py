import pandas as pd
import math
import plotly.express as px

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
url = "earthquakes.csv"

df_car = pd.read_csv(url,',')
print(df_car)

print(df_car.columns)
df_car = df_car[[ 'Year', 'Latitude', 'Longitude', 'Mag']]


# selecting rows based on condition 
df_car = df_car.loc[df_car['Year'] > 1800]
df_car = df_car.loc[(pd.isnull(df_car['Mag'])) == False]

    
print('\nResult dataframe :\n', 
      df_car)
# print(df_car)
# exit()

# Create a 3D figure
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('Year')
ax.set_ylabel('Longitude')
ax.set_zlabel('Latitude')

# Scatter plot the data with magnitude as color
ax.scatter(df_car['Year'], df_car['Longitude'], df_car['Latitude'], c=df_car['Mag'], cmap='viridis')
plt.savefig("mag-data.jpg")
# Show the plot
plt.show()






# Split the data into training and testing sets
train_size = int(len(df_car) * 0.8)
train_data = df_car.iloc[:train_size,:]
test_data = df_car.iloc[train_size:,:]

# Scale the data
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data[['Year', 'Longitude', 'Latitude', 'Mag']])
test_data_scaled = scaler.transform(test_data[['Year', 'Longitude', 'Latitude', 'Mag']])

# Create the training and testing datasets
def create_dataset(dataset, time_steps=1):
    X, y = [], []
    for i in range(len(dataset)-time_steps):
        X.append(dataset[i:(i+time_steps), :])
        y.append(dataset[i+time_steps, -1])
    return np.array(X), np.array(y)

time_steps = 3
X_train, y_train = create_dataset(train_data_scaled, time_steps)
X_test, y_test = create_dataset(test_data_scaled, time_steps)

# Create the LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)

# Make predictions
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

print(train_predict.shape)
exit()
# Inverse transform the scaled data
train_predict = scaler.inverse_transform(np.concatenate((X_train[:,:,1:],train_predict.reshape(-1,1)),axis=2))[:,:,-1]
test_predict = scaler.inverse_transform(np.concatenate((X_test[:,:,1:],test_predict.reshape(-1,1)),axis=2))[:,:,-1]

# Evaluate the model
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, test_predict)
print('Test MSE: %.3f' % mse)


exit()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df_car)
# print(scaled_data)

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]