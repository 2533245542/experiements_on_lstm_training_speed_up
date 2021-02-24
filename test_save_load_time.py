import unittest

from save_load_time import VariableSaverAndLoader

class TestVariableSaverAndLoader(unittest.TestCase):
    def test_basic(self):
        empty_variableSaverAndLoader = VariableSaverAndLoader(list_of_variables_to_save=None, save=False, load=False)
        assert empty_variableSaverAndLoader.duration_time_save == None
        assert empty_variableSaverAndLoader.start_time_save == None
        assert empty_variableSaverAndLoader.duration_time_load == None
        assert empty_variableSaverAndLoader.end_time_load == None
        assert empty_variableSaverAndLoader.list_of_loaded_variables == None
        assert empty_variableSaverAndLoader.load_is_successful == None
        assert empty_variableSaverAndLoader.save_is_successful == None

        a = 1
        b = 2
        c = {'a':a, 'b':b}
        save_variableSaverAndLoader = VariableSaverAndLoader(list_of_variables_to_save=[a,b,c], save=True, load=False)
        assert save_variableSaverAndLoader.duration_time_save != None
        assert save_variableSaverAndLoader.start_time_save != None
        assert save_variableSaverAndLoader.duration_time_load == None
        assert save_variableSaverAndLoader.end_time_load == None
        assert save_variableSaverAndLoader.list_of_loaded_variables == None
        assert save_variableSaverAndLoader.save_is_successful == True
        assert save_variableSaverAndLoader.load_is_successful == None




''''
    def test_time(self):
        VariableSaverAndLoader(list_of_variables_to_save=[number_of_days_to_predict_ahead, total_size, batch_size, inference_batch_size, input_length, number_of_input_variables, input_size, output_size, hidden_size, hyper_parameter_value_combination, train_input_output_tensor_list, test_input_output_tensor_list, many_to_many, loss_function, generate_train_prediction, generate_test_prediction, early_stopping], save=True).duration_time_save
        VariableSaverAndLoader(list_of_variables_to_save=[number_of_days_to_predict_ahead, total_size, batch_size, inference_batch_size, input_length, number_of_input_variables, input_size, output_size, hidden_size, hyper_parameter_value_combination, train_input_output_tensor_list, test_input_output_tensor_list, many_to_many, loss_function, generate_train_prediction, generate_test_prediction, early_stopping], load=True).duration_time_load

        VariableSaverAndLoader(list_of_variables_to_save=[precursor_dataset], save=True).duration_time_save
        VariableSaverAndLoader(list_of_variables_to_save=[precursor_dataset], load=True).duration_time_load

        VariableSaverAndLoader(list_of_variables_to_save=[dataloader], save=True).duration_time_save
        VariableSaverAndLoader(list_of_variables_to_save=[dataloader], load=True).duration_time_load
'''



