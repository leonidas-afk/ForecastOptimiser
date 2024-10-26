# @author Leonidas Lymberopoulos, Lamda Networks.

from numpy import array
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import pandas as pd
import os
import sys
import numpy as np
import numpy
import os
from flask import Flask,  Response, request
import json
import socket
import keras
from sklearn.preprocessing import MinMaxScaler


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def calculate_averages_based_on_Horizon(data, Horizon):

    print (data)
    data = data.flatten()
    print (data)
    X = list()
    print ("The data lenght is " + str(len(data)))
    print ("The Horizon is " + str(Horizon))

    match Horizon:
        case 2:
            print("Working with a Horizon of 2")
            number_of_horizons_to_average = int((len(data) - 2)/Horizon)
            X.append(data[0])

            for i in range(number_of_horizons_to_average):
                print ("Averaging horizon numbered " + str(i))
                current_line = data[ 1 + Horizon*(i): Horizon*(i+1) + 1 ]
                avg = sum(current_line) / len (current_line)
                X.append(avg)

            X.append(data[len(data)-1])

        case 3:
            print("Working with a Horizon of 3")
            if (len(data)==Horizon):
                X=data
            else:

                items_to_remove = Horizon*(Horizon-1)
                number_of_horizons_to_average = int( (len(data) - items_to_remove )/Horizon)
                print ("I will average # of " + str(number_of_horizons_to_average))

                X.append(data[0])

                print ("averaging numbers " + str(data[1]) + " and " + str(data[2]))
                X.append((data[1]+data[2])/2)

                for i in range(number_of_horizons_to_average):
                    print ("Averaging horizon numbered " + str(i))
                    current_line = data[ 3 + Horizon*(i): Horizon*(i+1) + 3 ]
                    avg = sum(current_line) / len (current_line)
                    X.append(avg)

                print ("averaging numbers " + str(data[-3]) + " and " + str(data[-2]))
                X.append((data[len(data)-3]+ data[len(data)-2])/2)

                X.append(data[len(data)-1])

        case 4:
            print("Working with a Horizon of 4")
            if (len(data)==Horizon):
                X=data

            else:
                items_to_remove = Horizon*(Horizon-1)
                number_of_horizons_to_average = int( (len(data) - items_to_remove )/Horizon)
                print ("I will average # of " + str(number_of_horizons_to_average))

                X.append(data[0])

                print ("averaging numbers " + str(data[1]) + " and " + str(data[2]))
                X.append((data[1]+data[2])/2)

                print ("averaging numbers " + str(data[3]) + " and " + str(data[4]) + " and " + str(data[5]))
                X.append((data[3]+data[4]+data[5])/3)

                for i in range(number_of_horizons_to_average):
                    print ("Averaging horizon numbered " + str(i))
                    current_line = data[ 6 + Horizon*(i): Horizon*(i+1) + 6 ]
                    avg = sum(current_line) / len (current_line)
                    X.append(avg)

                print ("averaging numbers " + str(data[-6]) + " and " + str(data[-5])+ " and " + str(data[-4]))
                X.append((data[len(data)-6]+ data[len(data)-5]+ data[len(data)-4])/3)

                print ("averaging numbers " + str(data[-3]) + " and " + str(data[-2]))
                X.append((data[len(data)-3]+ data[len(data)-2])/2)

                X.append(data[len(data)-1])
        case 5:
            print("Working with a Horizon of 5")
            if (len(data)==Horizon):
                X=data
            else:
                items_to_remove = Horizon*(Horizon-1)
                number_of_horizons_to_average = int( (len(data) - items_to_remove )/Horizon)

                print ("I will average # of " + str(number_of_horizons_to_average))

                X.append(data[0])

                print ("averaging numbers " + str(data[1]) + " and " + str(data[2]))
                X.append((data[1]+data[2])/2)

                print ("averaging numbers " + str(data[3]) + " and " + str(data[4]) + " and " + str(data[5]))
                X.append((data[3]+data[4]+data[5])/3)

                print ("averaging numbers " + str(data[6]) + " and " + str(data[7]) + " and " + str(data[8]) + " and " + str(data[9]))
                X.append((data[6]+data[7]+data[8]+data[9])/4)

                for i in range(number_of_horizons_to_average):
                    print ("Averaging horizon numbered " + str(i))
                    current_line = data[ 10 + Horizon*(i): Horizon*(i+1) + 10 ]
                    avg = sum(current_line) / len (current_line)
                    X.append(avg)

                print ("averaging numbers " + str(data[-10]) + " and " + str(data[-9]) + " and " + str(data[-8]) + " and " + str(data[-7]))
                X.append((data[-10]+data[-9]+data[-8]+data[-7])/4)

                print ("averaging numbers " + str(data[-6]) + " and " + str(data[-5])+ " and " + str(data[-4]))
                X.append((data[len(data)-6]+ data[len(data)-5]+ data[len(data)-4])/3)


                print ("averaging numbers " + str(data[-3]) + " and " + str(data[-2]))
                X.append((data[len(data)-3]+ data[len(data)-2])/2)

                X.append(data[len(data)-1])
        case 6:
            print("Working with a Horizon of 6")
            if (len(data)==Horizon):
                X=data
            else:
                items_to_remove = Horizon*(Horizon-1)
                number_of_horizons_to_average = int( (len(data) - items_to_remove )/Horizon)

                print ("I will average # of " + str(number_of_horizons_to_average))

                X.append(data[0])

                print ("averaging numbers " + str(data[1]) + " and " + str(data[2]))
                X.append((data[1]+data[2])/2)

                print ("averaging numbers " + str(data[3]) + " and " + str(data[4]) + " and " + str(data[5]))
                X.append((data[3]+data[4]+data[5])/3)

                print ("averaging numbers " + str(data[6]) + " and " + str(data[7]) + " and " + str(data[8]) + " and " + str(data[9]))
                X.append((data[6]+data[7]+data[8]+data[9])/4)

                print ("averaging numbers " + str(data[10]) + " and " + str(data[11]) + " and " + str(data[12]) + " and " + str(data[13]) +" and " + str(data[14]))
                X.append((data[10]+data[11]+data[12]+data[13]+data[14])/5)

                for i in range(number_of_horizons_to_average):
                    print ("Averaging horizon numbered " + str(i))
                    current_line = data[ 15 + Horizon*(i): Horizon*(i+1) + 15 ]
                    avg = sum(current_line) / len (current_line)
                    X.append(avg)

                print ("averaging numbers " + str(data[-15]) + str(data[-14]) + " and " + str(data[-13]) + " and " + str(data[-12]) + " and " + str(data[-11]))
                X.append((data[-15]+data[-14]+data[-13]+data[-12]+data[-11])/5)

                print ("averaging numbers " + str(data[-10]) + " and " + str(data[-9]) + " and " + str(data[-8]) + " and " + str(data[-7]))
                X.append((data[-10]+data[-9]+data[-8]+data[-7])/4)

                print ("averaging numbers " + str(data[-6]) + " and " + str(data[-5])+ " and " + str(data[-4]))
                X.append((data[len(data)-6]+ data[len(data)-5]+ data[len(data)-4])/3)


                print ("averaging numbers " + str(data[-3]) + " and " + str(data[-2]))
                X.append((data[len(data)-3]+ data[len(data)-2])/2)

                X.append(data[len(data)-1])
        case 7:
            print("Working with a Horizon of 7")
            if (len(data)==Horizon):
                X=data
            else:
                items_to_remove = Horizon*(Horizon-1)
                number_of_horizons_to_average = int( (len(data) - items_to_remove )/Horizon)

                print ("I will average # of " + str(number_of_horizons_to_average))

                X.append(data[0])

                print ("averaging numbers " + str(data[1]) + " and " + str(data[2]))
                X.append((data[1]+data[2])/2)

                print ("averaging numbers " + str(data[3]) + " and " + str(data[4]) + " and " + str(data[5]))
                X.append((data[3]+data[4]+data[5])/3)

                print ("averaging numbers " + str(data[6]) + " and " + str(data[7]) + " and " + str(data[8]) + " and " + str(data[9]))
                X.append((data[6]+data[7]+data[8]+data[9])/4)

                print ("averaging numbers " + str(data[10]) + " and " + str(data[11]) + " and " + str(data[12]) + " and " + str(data[13]) +" and " + str(data[14]))
                X.append((data[10]+data[11]+data[12]+data[13]+data[14])/5)

                print ("averaging numbers " + str(data[15]) + " and " + str(data[16]) + " and " + str(data[17]) + \
                    " and " + str(data[18]) +" and "+ str(data[19])+ " and "+str(data[20]))
                X.append((data[15]+data[16]+data[17]+data[18]+data[19]+data[20])/6)

                for i in range(number_of_horizons_to_average):
                    print ("Averaging horizon numbered " + str(i))
                    current_line = data[ 21 + Horizon*(i): Horizon*(i+1) + 21 ]
                    avg = sum(current_line) / len (current_line)
                    X.append(avg)

                print ("averaging numbers " + str(data[-21]) + str(data[-20]) + " and " + str(data[-19]) + " and " + str(data[-18]) + " and " + str(data[-17])+ " and " + str(data[-16]))
                X.append((data[-21]+data[-20]+data[-19]+data[-18]+data[-17]+data[-16])/6)

                print ("averaging numbers " + str(data[-15]) +" and "+ str(data[-14]) + " and " + str(data[-13]) + " and " + str(data[-12]) + " and " + str(data[-11]))
                X.append((data[-15]+data[-14]+data[-13]+data[-12]+data[-11])/5)

                print ("averaging numbers " + str(data[-10]) + " and " + str(data[-9]) + " and " + str(data[-8]) + " and " + str(data[-7]))
                X.append((data[-10]+data[-9]+data[-8]+data[-7])/4)

                print ("averaging numbers " + str(data[-6]) + " and " + str(data[-5])+ " and " + str(data[-4]))
                X.append((data[len(data)-6]+ data[len(data)-5]+ data[len(data)-4])/3)


                print ("averaging numbers " + str(data[-3]) + " and " + str(data[-2]))
                X.append((data[len(data)-3]+ data[len(data)-2])/2)

                X.append(data[len(data)-1])
        case _:
            print("Error 511: Not supported forecast horizon")

    print (X)

    return array(X)

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
		X.append(sequence[i:(i + batchSize)]) # configurable batchSize and configurable Horizon
		y.append(sequence[(i + batchSize):(i + batchSize + Horizon)])
		# if i <= (batchSize +1):
		# 	print(X)
		# 	print(y)
		# 	print()
	return array(X), array(y)


app = Flask(__name__)


@app.before_request
def log_request_info():
    app.logger.debug('Headers: %s', request.headers)
    app.logger.debug('Body: %s', request.get_data())


@app.route("/api/dynamic_algorithm_selection", methods=['POST'])
def predict_on_ANY():


    request.args.to_dict(flat=False)
    resolution = request.args.get('resolution')
    Horizon = request.args.get('Horizon')

    json_received = request.get_json(force=True)


    x_input = numpy.array(json_received['values'])

    n_steps = 4
    batchSize = 4
    n_features = 1 # 1 is the number of features

    from datetime import datetime
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    match resolution:
    	case "15m":
    		print("["+ dt_string +"]:" + " Resolution is set to 15 mins")
    		resolution_for_training = "15T"
    	case "30m":
    		print("["+ dt_string +"]:"+" Resolution is set to 30 mins")
    		resolution_for_training = "30T"
    	case "1h":
    		print("["+ dt_string +"]:" + "Resolution is set to 1 hour")
    		resolution_for_training = "1H"
    	case "2h":
    		print("["+ dt_string +"]:"+ "Resolution is set to 2 hours")
    		resolution_for_training = "2H"
    	case "4h":
    		print("["+ dt_string +"]:"+ "Resolution is set to 4 hours")
    		resolution_for_training = "4H"
    	case "8h":
    		print(+"["+ dt_string +"]:" "Resolution is set to 8 hours")
    		resolution_for_training = "8H"
    	case "1d":
    		print("["+ dt_string +"]:"+ " Resolution is set to 1 day")
    		resolution_for_training = "1D"
    	case "1w":
    		print("["+ dt_string +"]:"+ "Resolution is set to 1 week")
    		resolution_for_training = "7D"
    	case "1M":
    		print("["+ dt_string +"]:"+ " Resolution is set to 1 month")
    		resolution_for_training = "1M"
    	case _:
    		print("["+ dt_string +"]:"+ " Error 501: Not supported resolution") #need to send a message to the frontend

    print("["+ dt_string +"]:"+ " Forecasting Horizon is set to " + Horizon)

    Horizon = int(Horizon)
    if (Horizon>10) or (Horizon<1):
        print("["+ dt_string +"]:"+ " Error 502: Not supported Horizon") #need to send a message to the frontend


    batchSize=4

    algorithm = "LSTM"
    file_to_load = "./models/" + algorithm + resolution + "Horizon" + str(Horizon)+ ".keras"
    regressorLSTM = keras.models.load_model(file_to_load)

    algorithm = "RNN"
    file_to_load = "./models/" + algorithm + resolution + "Horizon" + str(Horizon)+ ".keras"
    regressorRNN = keras.models.load_model(file_to_load)

    algorithm = "GRU"
    file_to_load = "./models/" + algorithm + resolution + "Horizon" + str(Horizon)+ ".keras"
    regressorGRU = keras.models.load_model(file_to_load)


    goalsLSTM =0
    goalsRNN =0
    goalsGRU =0


    # Selecting g1 test values
    test_data = x_input


    ##################### STEP 1.1 = splitting input pattern ################################
    #split sequence acceps lists of numbers
    test_data_np = np.array(test_data)
    if Horizon<=1:
    	X_test, y_test = split_sequence(test_data_np, batchSize)
    else:
    	X_test, y_test = split_sequence_with_horizon(test_data_np, batchSize, Horizon)

    for i in range(len(X_test)):
    	print(X_test[i], y_test[i])
    ##################### STEP 1.2 = reshaping ################################
    n_features = 1
    if Horizon<=1:
    	# reshape from [samples, timesteps] into [samples, timesteps, features]
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], n_features))
        print("X_test :",X_test.shape,"y_test :",y_test.shape)
    else:
        print("X_test :",X_test.shape,"y_test :",y_test.shape)

    ##################### STEP 1.3 = Predicting ################################
    y_predictions_list = list()
    algorithms_list = list()
    rmse_list = list()

    for i in range(len(X_test)):
        current_line = X_test[i]
        current_line = current_line.reshape((1, n_steps, n_features))
        evaluationVar = y_test[i].reshape((1, Horizon))

        history_LSTM = regressorLSTM.evaluate(current_line, evaluationVar, verbose=1)
        rmse_LSTM = history_LSTM[2]
        history_RNN = regressorRNN.evaluate(current_line, evaluationVar, verbose=1)
        rmse_RNN = history_RNN[2]
        history_GRU = regressorGRU.evaluate(current_line, evaluationVar, verbose=1)
        rmse_GRU = history_GRU[2]
        from datetime import datetime

        now = datetime.now()

        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

        print ("["+ dt_string +"]:" + " Performance of LSTM, RNN and GRU in terms of RMSE for the pattern[" +str(i) +"] as received from the web UI")
        print ("["+ dt_string +"]:" +" LSTM achieved RMSE " + str(rmse_LSTM))
        print ("["+ dt_string +"]:" +" RNN achieved RMSE " + str(rmse_RNN))
        print ("["+ dt_string +"]:" +" GRU achieved RMSE " + str(rmse_GRU))

        if rmse_LSTM==min(rmse_LSTM, rmse_RNN, rmse_GRU):
            goalsLSTM = goalsLSTM +1
            y_predict = regressorLSTM.predict(current_line, verbose=1)
            y_predictions_list.append(y_predict)
            algorithms_list.append("LSTM")
            rmse_list.append(rmse_LSTM)
        if rmse_RNN==min(rmse_LSTM, rmse_RNN, rmse_GRU):
            goalsRNN = goalsRNN +1
            y_predict = regressorRNN.predict(current_line, verbose=1)
            y_predictions_list.append(y_predict)
            algorithms_list.append("RNN")
            rmse_list.append(rmse_RNN)
        if rmse_GRU==min(rmse_LSTM, rmse_RNN, rmse_GRU):
            goalsGRU = goalsGRU +1
            y_predict = regressorGRU.predict(current_line, verbose=1)
            y_predictions_list.append(y_predict)
            algorithms_list.append("GRU")
            rmse_list.append(rmse_GRU)



    # define Python dictionary
    result ={
  "predictions": y_predictions_list,
  "algorithm": algorithms_list,
  "rmse": rmse_list
}



    if (Horizon>1):

        predictions_array = numpy.array(y_predictions_list)
        y_predictions_list = calculate_averages_based_on_Horizon(predictions_array, Horizon)

    result= json.dumps({'predictions': y_predictions_list, 'algorithm': algorithms_list, 'rmse': rmse_list},
                       cls=NumpyEncoder)


    from datetime import datetime
    now = datetime.now()

    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

    print ("["+ dt_string +"]:" + " Performance of LSTM, RNN and GRU in terms of RMSE for the input testing dataset from the web UI")
    print ("["+ dt_string +"]:" +" LSTM achieved the least RMSE " + str(goalsLSTM) + " times")
    print ("["+ dt_string +"]:" +" RNN achieved the least RMSE " + str(goalsRNN) + " times")
    print ("["+ dt_string +"]:" +" GRU achieved the least RMSE " + str(goalsGRU) + " times")
    print ("["+ dt_string +"]:" +" Total batches tested were: " + str(len(X_test) ))

    print ("["+ dt_string +"]:"+"Printing the prediction array whose size is " + str(len(y_predictions_list)))
    for i in range(0,len(y_predictions_list)):
    	print (y_predictions_list[i])

    return Response(result, mimetype='application/json')


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Credentials', 'true')
    return response


if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0')
