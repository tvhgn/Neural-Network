from helpers import preprocess_data, data_generator

import numpy as np
import torch

class Layer():
    def __init__(self, n_input, nodes, activation, label):
        self.n_input = n_input
        self.nodes = nodes
        self.activation = activation
        self.label = label
        # Initializing weights and biases
        lbound = -1/np.sqrt(n_input)
        rbound = -lbound
        self.weights = torch.tensor(np.random.uniform(lbound, rbound, n_input*nodes).reshape(n_input, nodes), requires_grad=True)
        self.biases = torch.tensor(np.random.uniform(lbound, rbound, nodes).reshape(1, nodes), requires_grad=True)
        self.prev_step_w = torch.empty_like(self.weights) * 0
        self.prev_step_b = torch.empty_like(self.biases) * 0
    
    def __repr__(self):
        return f"{self.label} containing {self.nodes} nodes"
    
    def f_propogate(self, input_array):
        # Checking input array datatype
        if type(input_array) != torch.Tensor:
            input_array = torch.tensor(input_array)
        if input_array.dtype != torch.float64:
            input_array = input_array.type(torch.float64)
        # Calculate the linear output function 'z' using an input array and the weights/biases contained in this layer.
        z = torch.mm(input_array, self.weights) + self.biases
        # Calculate the activated output.
        if self.activation.lower() == "tanh":
            self.a = torch.tanh(z)
        elif self.activation.lower() == "relu":
            self.a = torch.relu(z)
        elif self.activation.lower() == "sigmoid":
            self.a = torch.sigmoid(z)
        elif self.activation.lower() == 'none':
            self.a = z
        else:
            raise RuntimeError("Activation function not recognized! Please try again.")
        
        return self.a