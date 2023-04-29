# Step 1: Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
# from keras.models import Dropout
from statsmodels.tsa.stattools import adfuller




# Step 2: Load the CSV data
df = pd.read_csv('reg_cluster1.csv', index_col= 'Year')
col = ["Average Magnitude"]
df.drop(col, axis=1, inplace=True)
df.plot(figsize=(12,6))
plt.savefig("graph.jpg")


# exit()
# df = df.dropna()
# print(df.head())
# exit()
# df['Average Magnitude'].fillna(4, inplace= True)

# def ad_test(dataset):
#      dftest = adfuller(dataset, autolag = 'AIC')
#      print("1. ADF : ",dftest[0])
#      print("2. P-Value : ", dftest[1])
#      print("3. Num Of Lags : ", dftest[2])
#      print("4. Num Of Observations Used For ADF Regression:",      dftest[3])
#      print("5. Critical Values :")
#      for key, val in dftest[4].items():
#          print("\t",key, ": ", val)
# print(df)

# df = df.diff()
# df.dropna(how='all', inplace = True)
# print(df)

# exit()
# ad_test(df)
# df.plot()
# plt.plot(df)
# plt.savefig("heyk.jpg")
# exit()

# print(df)
# exit()


#80% - 20% Split
train_size = int(len(df) * 0.8)
train_data, test_data = df.iloc[:train_size], df.iloc[train_size:]


# Step 6: Normalize the data
scaler = MinMaxScaler()
train_data = scaler.fit_transform(train_data)



n_input = 2
n_features = 1
generator = TimeseriesGenerator(train_data, train_data, length = n_input, batch_size = 1)


print(train_data)
# Step 7: Define the LSTM model architecture
model = Sequential()
model.add(LSTM(units=50, activation='relu', input_shape=(n_input, n_features)))
model.add(Dropout(0.2))
model.add(Dense(1))

# Step 8: Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# # Step 9: Train the model

history = model.fit(generator, epochs=50, batch_size=1, verbose=2)
loss_per_epoch = model.history.history['loss']
plt.plot(range(len(loss_per_epoch)), loss_per_epoch)
plt.savefig('loss.jpg')


test_prediction = []
last_train_batch = train_data[-n_input:] 
cur_batch = last_train_batch.reshape((1, n_input, 1))
#  n
for i in range (len(test_data)):
    cur_pred = model.predict(cur_batch)[0]
    test_prediction.append(cur_pred)
    cur_batch = np.append(cur_batch[:,1:,:], [[cur_pred]], axis=1)


# exit()
true_prediction = scaler.inverse_transform(test_prediction)
print(true_prediction)
# exit()
test_data['Prediction'] = true_prediction
print(test_data.head())
test_data.plot(figsize=(12,6))

plt.savefig("Graphs/eval_grah_reg1.jpg")
