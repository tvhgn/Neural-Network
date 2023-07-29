from layer import Layer
from helpers import generate_test_data

import numpy as np

class Network:
    def __init__(self, input_layer, structure=[]):
        self.structure = structure
        self.depth = len(structure)
        self.layers = []
        
        for i in range(self.depth):
            # Input layer
            if i == 0:
                self.layers.append(Layer(label=str(i+1), input_size=input_layer, num_nodes=structure[i],activation='relu'))
            # Output layer
            elif i == (self.depth-1):
                input_layer_size = len(self.layers[i-1].nodes)
                self.layers.append(Layer(label='output', input_size=input_layer_size, num_nodes=structure[i], activation='none'))
            else:
                input_layer_size = len(self.layers[i-1].nodes)
                self.layers.append(Layer(label=str(i+1), input_size=input_layer_size, num_nodes=structure[i], activation='relu'))
                
    def f_propogate(self, x_array):
        # List for contaiining predicted output of every row in x_array
        outputs = []
        # Iterate over rows. Each row contains one set of features that network will be trained on.
        for input_array in x_array:
            for layer in self.layers:
                # print(f"{layer.label} starts with input array:\n{input_array}")
                input_array = layer.f_propogate(input_array) # Output of one layer becomes input for next layer.
                # print(f"{layer.label} outputs:\n{input_array}\n")
            outputs.append(input_array) # Store predicted value for specific feature set in list.
        return np.asarray(outputs)

    def calc_loss(self, y_pred:list, y_true:list, function:str):
        if function == "mse":
            result = np.square(y_pred-y_true)
            return np.mean(result)
        else:
            raise RuntimeError("Loss function was not recognized, please try again.")