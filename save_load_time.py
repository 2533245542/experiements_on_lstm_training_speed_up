import pickle
from pathlib import Path
import time
'''
input: 
list_of_varaibles_to_save
save=Falas
load=False
file_name='some_file'

output:
list_of_loaded_variables=None
save_time
load_time
file_name

steps:
if save
    start_time_save = time.time()
    with open file:
        write
    end_time_save = time.time()
    
if load
    start_time_load = time.time()
    with open file:
        read
    end_time_load = time.time()
    
tests:
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

'''

class VariableSaverAndLoader():
    def __init__(self, list_of_variables_to_save=None, save=False, load=False, file_name='data/target_file.dat',
                 lazy=False):
        # inputs
        self.list_of_variables_to_save = list_of_variables_to_save
        self.save = save
        self.load = load
        self.file_name = file_name
        self.lazy = lazy

        # outputs
        self.start_time_save = None
        self.save_is_successful = None
        self.end_time_save = None
        self.duration_time_save = None

        self.start_time_load = None
        self.load_is_successful = None
        self.end_time_load = None
        self.duration_time_load = None

        self.list_of_loaded_variables = None

        if not self.lazy:
            if self.save:
                self.start_time_save = time.time()
                self.save_is_successful = self.do_save(self.file_name, self.list_of_variables_to_save)
                self.end_time_save = time.time()
                self.duration_time_save = self.end_time_save - self.start_time_save

            if self.load:
                self.start_time_load = time.time()
                self.load_is_successful, self.list_of_loaded_variables = self.do_load(self.file_name)
                self.end_time_load = time.time()
                self.duration_time_load = self.end_time_load - self.start_time_load

    def do_save(self, file_name, list_of_variables_to_save):
        parent_path = Path(file_name).parent
        parent_path.mkdir(parents=True, exist_ok=True)
        with open(file_name, "wb") as f:
            pickle.dump(list_of_variables_to_save, f)
        return True  # success

    def do_load(self, file_name):
        with open(file_name, "rb") as f:
            list_of_loaded_variables = pickle.load(f)
        return True, list_of_loaded_variables


