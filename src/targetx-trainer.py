# This is to verify how it works in terms of batches
# https://www.kaggle.com/code/ritesh7355/develop-lstm-models-for-time-series-forecasting
#
# edw na typwsw 4 ades me train data me tis times tous kai meta na rwthsw na
# dw kata poso ta kserei.
#
#
from numpy import array
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras
import pandas as pd
import os
import sys
import numpy as np
import numpy

import matplotlib.pyplot as plt
import math


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps
		# check if we are beyond the sequence
		if end_ix > len(sequence)-1:
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

def split_sequence_with_horizon(sequence, batchSize, Horizon):
	X, y = list(), list()
	number_of_batches_to_train_or_test = len(sequence) - batchSize - Horizon -1
	for i in range(number_of_batches_to_train_or_test):
		print ("Using training or test batch numbered " + str(i))
		# X.append(sequence[i:(i + batchSize), 0]) # configurable batchSize and configurable Horizon
		# y.append(sequence[(i + batchSize):(i + batchSize + Horizon), 0])
		X.append(sequence[i:(i + batchSize)]) # configurable batchSize and configurable Horizon
		y.append(sequence[(i + batchSize):(i + batchSize + Horizon)])
		# if i <= (batchSize +1):
		# 	print(X)
		# 	print(y)
		# 	print()
	return array(X), array(y)


if "RESOLUTION" in os.environ:
	resolution = os.environ['RESOLUTION']
	print ("The environment variable RESOLUTION is set to " + resolution)

print ("resolution = " + resolution)

if "HORIZON" in os.environ:
	Horizon = os.environ['HORIZON']
	print ("The environment variable HORIZON is set to " + Horizon)
	Horizon = int(Horizon)

# if "BATCH_SIZE" in os.environ:
# 	batchSize = int(os.environ['BATCH_SIZE'])
# 	print ("The environment variable BATCH_SIZE is set to " + str(batchSize))

batchSize=4
print ("batch size = " + str(batchSize))

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Add the script directory to the Python path
sys.path.append(script_dir)

directory = './powerData1/' #this has only one file, poweData1 is the one that has all the data.
files = os.listdir(directory)
# Initialize an empty list to store dataframes
dfs = []

# Iterate over each file
for file in files:
    # Check if the file is a CSV file
    if file.endswith('.csv'):
        # Construct the full path to the CSV file

        file_path = os.path.join(directory, file)

        # Specify the columns you want to read
        columns_to_read = ['Datum', 'Zeit', 'g1']

        # Read the CSV file into a pandas DataFrame
        df = pd.read_csv(file_path, header=4, delimiter=';', engine='python', usecols=columns_to_read)

        # Append the DataFrame to the list
        dfs.append(df)
del df
# Concatenate all DataFrames into a single DataFrame
df = pd.concat(dfs, ignore_index=True)
del dfs


# # Combine date and time columns into a single datetime column -> I want this to be an SQL date
df['Datetime'] = pd.to_datetime(df['Datum'] + ' ' + df['Zeit'], format='%d.%m.%Y %H:%M')
#
# # Drop the separate date and time columns if needed
# df.drop(['Datum', 'Zeit'], axis=1, inplace=True)
#
# # Sort df by a specific column and drop index
# df = df.sort_values(by='Datetime', ascending=True)
# df.reset_index(drop=True, inplace=True)
# df = df[['Datetime', 'g1']]

# edw kanei merge o ilithios

# # Create a DataFrame with the complete range of datetime values
# start_date = df['Datetime'].min()
# end_date = df['Datetime'].max()
# complete_range = pd.date_range(start=start_date, end=end_date, freq='min')
# complete_df = pd.DataFrame({'Datetime': complete_range})
#
# # Merge the complete DataFrame with your existing DataFrame
# df = pd.merge(complete_df, df, on='Datetime', how='left')

df.fillna(0)

print (df)

#################################################
#
#     Using the interpolate method to fill the missing values
#################################################
df_interopolated  = df.interpolate(method ='linear', limit_direction ='forward')
#df.interpolate(method ='linear', limit_direction ='backward', limit = 1)
df_interopolated.fillna(0)
#################################################
#
#     Using resampling
#Resampling interval such as 15T for 15 minutes (H is for hour, D is for days, M is for months)
#################################################


#print (df)
resolution_for_training="0m"

print("Now matching OS resolution")
print (resolution)
match resolution:
	case "1m":
		print("Resolution is set to 1 min")
		resolution_for_training = "1T"
	case "15m":
		print("Resolution is set to 15 mins")
		resolution_for_training = "15T"
	case "30m":
		print("Resolution is set to 30 mins")
		resolution_for_training = "30T"
	case "1h":
		print("Resolution is set to 1 hour")
		resolution_for_training = "1H"
	case "2h":
		print("Resolution is set to 2 hours")
		resolution_for_training = "2H"
	case "4h":
		print("Resolution is set to 4 hours")
		resolution_for_training = "4H"
	case "8h":
		print("Resolution is set to 8 hours")
		resolution_for_training = "8H"
	case "1d":
		print("Resolution is set to 1 day")
		resolution_for_training = "1D"
	case "1w":
		print("Resolution is set to 1 week")
		resolution_for_training = "7D"
	case "1M":
		print("Resolution is set to 1 month")
		resolution_for_training = "1M"
	case _:
		print("Error 510: Not supported resolution") #need to send a message to the frontend

print("Resolution is set to " + resolution_for_training)


# df_resampled  = df_interopolated.resample(resolution_for_training, on ='Datetime').mean()
# df_resampled.fillna(0) #to deal with the very important problem of na values

df_interopolated.drop(['Datum', 'Zeit'], axis=1, inplace=True)
df_interopolated.info()


df_interopolated_Indexed = df_interopolated.set_index(pd.DatetimeIndex(df_interopolated["Datetime"]))
#df_interopolated_Indexed = df.set_index(pd.DatetimeIndex(df_interopolated["Datetime"])).drop("Datetime", axis =1)
# print (df_interopolated_Indexed.head())
# df_interopolated_Indexed.info()


#Here I do the Resampling

# df_resampled = df_interopolated
# df_resampled.set_index('Datetime', drop=False).resample(resolution_for_training,on ="Datetime").mean()

# ask Leo!

df_resampled  = df_interopolated_Indexed.resample(resolution_for_training,on ="Datetime").mean()
#df_resampled = df_resampled.set_index(pd.DatetimeIndex(df_resampled["Datetime"]))
#df_resampled  = df_interopolated_Indexed.resample(resolution_for_training, on ="Datetime").mean()
#df_resampled  = df_interopolated_Indexed.resample(resolution_for_training).mean()

# ask Leo: first I do interpolatioin then I fill any missing values left
df_resampled  = df_resampled.interpolate(method ='linear', limit_direction ='forward')
df_resampled.fillna(0) #to deal with the very important problem of na values


# the index of resampled is the timestamps
# the value of the sigle column resampled is the g1 value
print (df_resampled.head())
print ("---------RESAMPLE INDEX----------")
print(df_resampled.index)
print ("--------------------")
df_resampled.info()

# raw_seq_full_data = df_resampled['g1'].to_numpy()
# raw_seq_non_rounded=raw_seq_full_data
# raw_seq = np.round(raw_seq_non_rounded, decimals=3)

all_data_with_dates = df_resampled

# THIS IS THE DATA THAT I GIVE TO VANGELIS.
#data_file = "./data_in_csv/" + algorithm + resolution +".csv"
#pd.DataFrame(train_data_with_dates).to_csv(train_data_file, parse_dates=['Datetime'])
#pd.DataFrame(all_data_with_dates).to_csv(data_file,index_label=False,header=None)



########################################################################
#       Continuing the USA code
#		Using different batch sizes instead of batch size fixed to 50
#       The var is the batchSize
########################################################################

data = df_resampled

# Setting 80 percent data for training
training_data_len = math.ceil(len(data) * .8)
print (training_data_len)

#Splitting the dataset
train_data = data[:training_data_len]
test_data = data[training_data_len:]
print(train_data.shape, test_data.shape)

# Selecting g1 values
dataset_train = train_data
# Reshaping 1D to 2D array
dataset_train = np.reshape(dataset_train, (-1,1))
dataset_train.shape

###############################################################################
#
#
#				my method with plain numbers
#
#
###############################################################################

print ("******** GOING WITH THE SEE NUMBERS OPTION *****************")

n_features = 1
# print (train_data)


##################### STEP 1 = splitting ################################
#split sequence acceps lists of numbers
train_data_np = np.array(train_data)
if Horizon<=1:
	X_train, y_train = split_sequence(train_data_np, batchSize)
else:
	X_train, y_train = split_sequence_with_horizon(train_data_np, batchSize, Horizon)

# for i in range(len(X_train)):
# 	print(X_train[i], y_train[i])

##################### STEP 2 = reshaping ################################

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
if Horizon<=1:
	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], n_features))
else:
	X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))
	y_train = np.reshape(y_train, (y_train.shape[0],Horizon))
	print("X_train :",X_train.shape,"y_train :",y_train.shape)

##################### STEP 3 = training ################################

# importing libraries
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Dropout
from keras.layers import GRU, Bidirectional
from keras.optimizers import SGD
from sklearn import metrics
from sklearn.metrics import mean_squared_error

algorithm="LSTM"
file_to_save = "./models/" + algorithm + resolution + "Horizon" + str(Horizon)+ ".keras"

#Initialising the model
regressorLSTM = Sequential()

regressorLSTM.add(LSTM(50, activation='relu', input_shape=(batchSize, 1)))
regressorLSTM.add(Dense(Horizon))

#Compiling the model
regressorLSTM.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy", "root_mean_squared_error", "f1_score"])

#Fitting the model
regressorLSTM.fit(X_train, y_train, epochs = 5, batch_size = 4)
regressorLSTM.summary()
regressorLSTM.save(file_to_save)
history_LSTM = regressorLSTM.evaluate(X_train,y_train,verbose=0)
rmse_LSTM = history_LSTM[2]
accuracy_LSTM=history_LSTM[1]


###############################################################################################################
algorithm = "RNN"
file_to_save = "./models/" + algorithm + resolution + "Horizon" + str(Horizon)+ ".keras"

regressorRNN = Sequential()
regressorRNN.add(SimpleRNN(50, activation='relu', input_shape=(batchSize, 1)))
regressorRNN.add(Dense(Horizon))

#Compiling the model
regressorRNN.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy", "root_mean_squared_error", "f1_score"])

# fitting the model
regressorRNN.fit(X_train, y_train, epochs = 5, batch_size = 4)
regressorRNN.summary()
regressorRNN.save(file_to_save)
history_RNN = regressorRNN.evaluate(X_train,y_train,verbose=0)
rmse_RNN = history_RNN[2]
accuracy_RNN=history_RNN[1]

###############################################################################################################

################################################################################
# Initialising the model
algorithm = "GRU"
file_to_save = "./models/" + algorithm + resolution + "Horizon" + str(Horizon)+ ".keras"
regressorGRU = Sequential()

regressorGRU.add(GRU(50, activation='relu', input_shape=(batchSize, 1)))
regressorGRU.add(Dense(Horizon))

#Compiling the model
regressorGRU.compile(optimizer = 'adam',
					loss = 'mean_squared_error',
					metrics = ["accuracy", "root_mean_squared_error", "f1_score"])

# Fitting the data
regressorGRU.fit(X_train,y_train,epochs=5,batch_size=4)
regressorGRU.summary()
regressorGRU.save(file_to_save)
history_GRU = regressorGRU.evaluate(X_train,y_train,verbose=0)
rmse_GRU = history_GRU[2]
accuracy_GRU=history_GRU[1]
################################################################################

#################### Step 3: prediction #################################
# predictions with X_test data

# -601.18      ]
#  [-132.70666667]
#  [-132.70666667]
#  [-132.70666667
# -10.9 ]
#  [-10.68]
#  [-10.68]
#  [-17.92 -> -10.61333333 -> -20 edwse.

# #-34.94166667]
#  [-11.64666667]
#  [-11.37333333]
#  [-15.18      ]] [-10.92] edwse
#  [-15.18      ]] [-10.92] edwse [-11.717098
# #
#
# x_input = array([34.94166667 , -11.64666667, -11.37333333, -15.18])
# x_input = x_input.reshape((1, batchSize, n_features))
# y_RNN = model.predict(x_input)


#forecasting all my testing data set
# x_input = np.array(test_data)
# x_input = x_input.reshape((1, batchSize, n_features))
# y_RNN = model.predict(x_input)



##################### STEP 3.1 = splitting ################################
#split sequence acceps lists of numbers
test_data_np = np.array(test_data)
if Horizon<=1:
	X_test, y_test = split_sequence(test_data_np, batchSize)
else:
	X_test, y_test = split_sequence_with_horizon(test_data_np, batchSize, Horizon)

# for i in range(len(X_test)):
# 	print(X_test[i], y_test[i])
##################### STEP 3.2 = reshaping ################################
if Horizon<=1:
	# reshape from [samples, timesteps] into [samples, timesteps, features]
	n_features = 1
	X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
else:
	X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))
	y_test = np.reshape(y_test, (y_test.shape[0],Horizon))
	print("X_test :",X_test.shape,"y_test :",y_test.shape)

##################### STEP 3.3 = Predicting ################################
y_LSTM = regressorLSTM.predict(X_test)
y_RNN = regressorRNN.predict(X_test)
y_GRU = regressorGRU.predict(X_test)
# print ("The predictions are: ")
# for i in range(len(y_LSTM)):
# 	print(y_LSTM[i])

# ##################### STEP 3.4= Plotting ################################
#
# if Horizon <=1:
#
# 	fig, axs = plt.subplots(3,figsize =(18,12),sharex=True, sharey=True)
# 	fig.suptitle('Model Predictions')
#
# 	#Plot for LSTM predictions
# 	axs[0].plot(train_data.index[:], train_data[:], label = "train_data", color = "blue")
# 	axs[0].plot(test_data.index, test_data, label = "test_data", linestyle='--', marker='o', color = "green")
# 	axs[0].plot(test_data.index[batchSize:], y_LSTM, label = "y_LSTM", linestyle='--', marker='x', color = "red")
# 	axs[0].legend()
#
# 	#Plot for RNN predictions
# 	axs[1].plot(train_data.index[:], train_data[:], label = "train_data", color = "blue")
# 	axs[1].plot(test_data.index, test_data, label = "test_data", linestyle='--', marker='o', color = "green")
# 	axs[1].plot(test_data.index[batchSize:], y_RNN, label = "y_RNN", linestyle='--', marker='x', color = "red")
# 	axs[1].legend()
#
# 	#Plot for GRU predictions
# 	axs[2].plot(train_data.index[:], train_data[:], label = "train_data", color = "blue")
# 	axs[2].plot(test_data.index, test_data, label = "test_data", linestyle='--', marker='o', color = "green")
# 	axs[2].plot(test_data.index[batchSize:], y_GRU, label = "y_GRU", linestyle='--', marker='x', color = "red")
# 	axs[2].legend()
#
# 	# plt.xlabel("Time")
# 	# plt.ylabel("g1")
# 	# algorithm="ALL"
# 	plt.show()
# 	# file_to_save = "./figures/" + algorithm + resolution + str(batchSize)+".png"
# 	# plt.savefig(file_to_save)
# else:
# 	y1,y2 = np.array_split(y_LSTM, 2, axis=1)
# 	#y= np.reshape(y_RNN_O,[-1])
#
# 	# y_RNN_O_left = np.reshape(y_RNN_O.shape[0])
# 	# y_RNN_O_right = np.reshape(y_RNN_O.shape[1])
# 	#
# 	# print(y_RNN_O_left.shape)
# 	# print(y_RNN_O_right.shape)
# 	#
# 	#print(y.shape)
# 	print(y_LSTM)
# 	print(y1)
# 	print(y2)
#
# 	fig, axs = plt.subplots(3,figsize =(18,12),sharex=True, sharey=True)
# 	fig.suptitle('Model Predictions')
#
# 	#SOS is the batchSize + Horizon + 1. This derives from the number of blocks defined earlier! See line 272
#
# 	test_data_len = len(test_data)
#
# 	print("The length of test data is " + str(test_data_len))
# 	print("The length of predictions is " + str(len(y1)))
#
#
# 	#Plot for RNN predictions, I need two prediction points per batch
# 	axs[0].plot(train_data.index[:], train_data[:], label = "train_data", color = "blue")
# 	axs[0].plot(test_data.index, test_data, label = "test_data", linestyle='--', marker='o', color = "green")
# 	#axs[0].plot(test_data.index, y_RNN_O, label = "y_RNN", color = "brown")
# 	axs[0].plot(test_data.index[batchSize:test_data_len - Horizon-1], y1, label = "y_LSTM_1stValue", linestyle='--', marker='x', color = "red")
# 	#axs[0].plot(test_data.index[batchSize+Horizon+1:], y1, label = "y_RNN_1stValue", color = "orange")
# 	axs[0].legend()
#
#
# 	axs[1].plot(train_data.index[:], train_data[:], label = "train_data", color = "blue")
# 	axs[1].plot(test_data.index, test_data, label = "test_data", linestyle='--', marker='o', color = "green")
# 	axs[1].plot(test_data.index[batchSize+1:test_data_len - Horizon], y2, label = "y_LSTM_2ndValue", linestyle='--', marker='x', color = "red")
# 	axs[1].legend()
# 	#
# 	# #Plot for LSTM predictions
# 	# axs[1].plot(train_data.index[batchSize*3:], train_data[batchSize*3:], label = "train_data", color = "b")
# 	# axs[1].plot(test_data.index, test_data, label = "test_data", color = "g")
# 	# axs[1].plot(test_data.index[batchSize:], y_LSTM_O, label = "y_LSTM", color = "orange")
# 	# axs[1].legend()
# 	# axs[1].title.set_text("LSTM with batch size " + str(batchSize) + " Accuracy=" + str(accuracy_LSTM)+ " RMSE="+ str(rmse_LSTM))
# 	#
# 	# #Plot for GRU predictions
# 	# axs[2].plot(train_data.index[batchSize*3:], train_data[batchSize*3:], label = "train_data", color = "b")
# 	# axs[2].plot(test_data.index, test_data, label = "test_data", color = "g")
# 	# axs[2].plot(test_data.index[batchSize:], y_GRU_O, label = "y_GRU", color = "red")
# 	# axs[2].legend()
# 	# axs[2].title.set_text("GRU with batch size " + str(batchSize) + " Accuracy=" + str(accuracy_GRU)+ " RMSE="+ str(rmse_GRU))
#
# 	plt.xlabel("Time")
# 	plt.ylabel("g1")
# 	algorithm="ALL"
# 	plt.show()
