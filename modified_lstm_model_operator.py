import torch

class ModifiedLSTMModelOperator():
    def __init__(self, hyper_parameter_value_combination, train_input_output_tensor_list, test_input_output_tensor_list, many_to_many, loss_function, generate_train_prediction=False, generate_test_prediction=True, early_stopping=False, early_stopping_patience=7, early_stopping_delta=0, customized_model_layer_name_to_layer_dictionary=None, batch_data_generator=None, device=None):

        self.hyper_parameter_value_combination = hyper_parameter_value_combination
        self.train_input_output_tensor_list = train_input_output_tensor_list
        self.test_input_output_tensor_list = test_input_output_tensor_list
        self.many_to_many = many_to_many
        self.loss_function = loss_function
        self.generate_train_prediction = generate_train_prediction
        self.generate_test_prediction = generate_test_prediction
        self.early_stopping = early_stopping
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_delta = early_stopping_delta
        self.customized_model_layer_name_to_layer_dictionary = customized_model_layer_name_to_layer_dictionary
        self.batch_data_generator = batch_data_generator
        self.device = device

        self.number_of_input_variables = 3
        self.number_of_output_variables = 1

        if True:
            self.model = LSTM(number_of_input_variables=self.number_of_input_variables, number_of_hidden_dimensions=self.hyper_parameter_value_combination[ 'number_of_hidden_dimensions'], number_of_output_variables=self.number_of_output_variables, dropout_rate=self.hyper_parameter_value_combination['dropout_rate'], customized_model_layer_name_to_layer_dictionary=self.customized_model_layer_name_to_layer_dictionary)
            self.model = self.model.to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hyper_parameter_value_combination['learning_rate'])
            self.batch_data_generator = batch_data_generator
            self.list_of_epoch_loss = self.train_model(batch_data_generator)

    def train_model(self, batch_data_generator):
        '''
        input: dataset
        output: time it takes to finish

        '''
        self.model.train()
        list_of_epoch_loss = []  # records the accumulated loss in each epoch, refreshed after each epoch ends
        for i in range(self.hyper_parameter_value_combination['number_of_training_epochs']):
            epoch_loss = 0
            for input_of_batch_data, output_of_batch_data in batch_data_generator:  # train and test input tensor could be a batch of data in future implementation
                # NEW turn into pytorch tensor
                input_of_batch_data = torch.Tensor(input_of_batch_data.numpy()).to(self.device)
                output_of_batch_data = torch.Tensor(output_of_batch_data.numpy()).reshape(-1,1,1).to(self.device)

                self.optimizer.zero_grad()

                h_0, c_0 = self.model.init_hidden(number_of_sequences_in_a_batch=input_of_batch_data.shape[0])  # batch size
                h_0 = h_0.to(self.device)
                c_0 = c_0.to(self.device)

                predicted_train_output_tensor, _ = self.model(input_tensor=input_of_batch_data, tuple_of_h_0_c_0=( h_0, c_0))  # expect input of shape (batch_size, seq_len, input_size)
                if self.many_to_many:
                    loss_of_a_batch = self.loss_function(predicted_train_output_tensor, output_of_batch_data)  # note that batch size is always 1
                else:
                    loss_of_a_batch = self.loss_function(predicted_train_output_tensor[:, -1:, :], output_of_batch_data)  # note that batch size is always 1
                loss_of_a_batch.backward()
                self.optimizer.step()
                epoch_loss += loss_of_a_batch.item()

            list_of_epoch_loss.append(epoch_loss)
            print(f'epoch: {i:3} loss: {epoch_loss:10.8f}')

        return list_of_epoch_loss



import torch
from torch import nn
class LSTM(nn.Module):
    '''We create an LSTM model consisting of LSTM cells and a linear layer. The hidden state from the last LSTM cell and time step will be feed into the linear layer; the linear layer performs linear transformation for the hidden state and outputs the prediction.

    This LSTM class has two functions: 1. create a model 2. specify how the model handle the inputs

    As of 1., the model is fixed to have three layers (lstm, dropout, linear), and this class provides a default way of creating each of the layer; however, users can customized each layer by providing a layer_name_to_layer_dictionary.

    '''

    def __init__(self, number_of_input_variables=1, number_of_hidden_dimensions=100, number_of_output_variables=1, dropout_rate=0.5, customized_model_layer_name_to_layer_dictionary=None):
        super().__init__()

        self.number_of_input_variable = number_of_input_variables
        self.number_of_hidden_dimensions = number_of_hidden_dimensions
        self.output_size = number_of_output_variables
        self.dropout_rate = dropout_rate
        self.customized_model_layer_name_to_layer_dictionary = customized_model_layer_name_to_layer_dictionary

        if self.customized_model_layer_name_to_layer_dictionary == None:
            # default layers
            self.lstm = nn.LSTM(input_size=number_of_input_variables, hidden_size=number_of_hidden_dimensions, batch_first=True)  # expect input of shape (batch, seq_len, input_size)
            self.dropout = nn.Dropout(dropout_rate)
            self.linear = nn.Linear(number_of_hidden_dimensions, number_of_output_variables)
        else:
            # customized model in the form of layers
            self.lstm = self.customized_model_layer_name_to_layer_dictionary['lstm']
            self.dropout = self.customized_model_layer_name_to_layer_dictionary['dropout']
            self.linear = self.customized_model_layer_name_to_layer_dictionary['linear']


    def init_hidden(self, number_of_sequences_in_a_batch):
        return (torch.zeros(1, number_of_sequences_in_a_batch, self.number_of_hidden_dimensions), torch.zeros(1, number_of_sequences_in_a_batch, self.number_of_hidden_dimensions))  #  num_layers, batch size (number of sequences), hidden_size

    def forward(self, input_tensor, tuple_of_h_0_c_0):
        '''
        :param input_tensor: an array where the length of each sequence and the number of sequences in a batch explained in the other arguments
        '''

        h_0, c_0 = tuple_of_h_0_c_0
        lstm_out, (h_0, c_0) = self.lstm(input_tensor, (h_0, c_0))  # expected input of shape (batch_size, seq_len, input_size)
        lstm_out_dropout = self.dropout(lstm_out)
        predictions = self.linear(lstm_out_dropout)  # expected input of shape (batch_size, any number of dimensions*, hidden_dim)
        return predictions, (h_0, c_0)