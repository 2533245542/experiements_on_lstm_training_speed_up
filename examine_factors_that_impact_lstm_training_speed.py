import torch
import time
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from torch import nn
from modified_lstm_model_operator import ModifiedLSTMModelOperator

device = torch.device("cpu")  # TODO cpu or gpu
# device = torch.device("cuda")

# inputs
number_of_days_to_predict_ahead = 1
total_size = 3000
batch_size = 160  # TODO 1, 20, 40, 80, 160
input_length = 20
number_of_input_variables = 3
input_size = 3
output_size = 1
hidden_size = 30

hyper_parameter_value_combination = {'dropout_rate': 0.0, 'learning_rate': 0.001, 'number_of_hidden_dimensions': 30, 'number_of_training_epochs': 30}
train_input_output_tensor_list = None
test_input_output_tensor_list = None
many_to_many = False
loss_function = nn.L1Loss()
generate_train_prediction = False
generate_test_prediction = False
early_stopping = False

# outputs
train_time = 0.0

# steps
## create  dataframe
case = pd.Series(list(range(1, 1 + total_size)))
call = case * 3 + 10
case_call_sum = case + call
precursor_dataset = pd.DataFrame({'case': case.values.astype(np.float64), 'call': call.values.astype(np.float64), 'case_call_sum': case_call_sum.values.astype(np.float64)})

## create dataset and dataloader
start_time_data_processing = time.time()
dataloader = tf.keras.preprocessing.timeseries_dataset_from_array(data=precursor_dataset.values[:-input_length, :], targets=precursor_dataset.values[input_length:, 0], sequence_length=input_length, batch_size=batch_size)
end_time_data_processing = time.time()
duration_time_data_processing = (end_time_data_processing - start_time_data_processing)/60
print('data processing takes time: ' + str(duration_time_data_processing))

## make operator
start_time_model_training = time.time()
modifiedLSTMModelOperator = ModifiedLSTMModelOperator(hyper_parameter_value_combination=hyper_parameter_value_combination, train_input_output_tensor_list=train_input_output_tensor_list, test_input_output_tensor_list=test_input_output_tensor_list, many_to_many=many_to_many, loss_function=loss_function, generate_train_prediction=generate_train_prediction, generate_test_prediction=generate_test_prediction, early_stopping=early_stopping, batch_data_generator=dataloader, device=device)
end_time_model_training = time.time()
duration_time_model_training = (end_time_model_training - start_time_model_training)/60
print('model training takes time: ' + str(duration_time_model_training))

# when batch_size=1, batch generator has a size of 2961
# when batch_size=80, ............................ 38; all batches have size of 80, except last batch has size of 1
