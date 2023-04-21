# Step 1: Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

# Step 2: Load the CSV data
df = pd.read_csv('reg_data.csv', index_col= 'Year')

train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]


# Step 6: Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)




n_input = 4
generator = TimeseriesGenerator(train_data, train_data, length = n_input, batch_size = 1)



# Step 7: Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_input, 1)))
model.add(Dense(units=1))

# Step 8: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# # Step 9: Train the model

history = model.fit(generator, epochs=100, batch_size=1, verbose=2)
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.savefig('loss.jpg')


test_prediction = []
last_train_batch = train_data[-n_input:] 
cur_batch = last_train_batch.reshape((1, n_input, 1))

for i in range (len(test_data)):
    cur_pred = model.predict(cur_batch)[0]
    test_prediction.append(cur_pred)
    cur_batch = np.append(cur_batch[:,1:,:], [[cur_pred]], axis=1)

true_prediction = scaler.inverse_transform(test_prediction)
test_data['Prediction'] = true_prediction
print(test_data.head())
test_data.plot(figsize=(12,6))

plt.savefig("evaluation-graph.jpg")
