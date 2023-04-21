import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

weather_data = pd.read_csv('reg_data.csv', index_col='Year', parse_dates=True)

plt.figure(figsize=(10, 6))
plt.plot(weather_data)
plt.title('Weather Data')
plt.xlabel('Date')
plt.ylabel('Temperature')
plt.savefig("output.jpg")
plt.show()
# print(weather_data)
# exit()
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(weather_data)
print(scaled_data)
exit()
# print(scaled_data)

train_size = int(len(scaled_data) * 0.8)
test_size = len(scaled_data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

def create_dataset(data, look_back=1):
    X, Y = [], []
    for i in range(len(data)-look_back-1):
        X.append(data[i:(i+look_back), 0])
        Y.append(data[i+look_back, 0])
    return np.array(X), np.array(Y)

look_back = 7
X_train, Y_train = create_dataset(train_data, look_back)
X_test, Y_test = create_dataset(test_data, look_back)

# print(X_train.shape)
# exit(Y_train.shape)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)
# exit()

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform([Y_test])
rmse = np.sqrt(np.mean(((predictions - Y_test) ** 2)))
print(f'RMSE: {rmse}')

plt.figure(figsize=(10, 6))
plt.plot(Y_test.reshape(-1), label='Actual')
plt.plot(predictions.reshape(-1), label='Predicted')
plt.title('Weather Prediction')
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.legend()
plt.savefig("output1.jpg")
plt.show()
