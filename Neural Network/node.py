import numpy as np

class Node:
    """
    Represents a node in a neural network.

    Attributes:
        input_size (int): The size of the input array.
        activation (str): The activation function of the node.
        weights (array): The weights of the node's connections.
        bias (float): The bias of the node.

    Methods:
        compute_output(input_array):
            Compute the output of the node for the given input array.

    """

    def __init__(self, input_size, activation):
        """
        Initializes a Node object.

        Parameters:
            input_size (int): The size of the input array.
            activation (str): The activation function of the node.

        """
        self.input_size = input_size
        self.activation = activation
        self.weights = np.random.uniform(-1, 1, input_size)  # Weights initialization
        self.bias = np.random.uniform(-1, 1, 1)  # Bias initialization

    def compute_output(self, input_array):
        """
        Compute the output of the node for the given input array.

        Parameters:
            input_array (array): The input array.

        Returns:
            output (float): The output of the node.

        Raises:
            RuntimeError: If the activation function is not 'relu' or 'none'.

        """
        # General linear transformation output = sum of all x_i*w_i + b combinations
        lin_trans = round(np.sum(np.multiply(input_array, self.weights) + self.bias), 4)
        # Relu activation function
        if self.activation == 'relu':
            return max(0, lin_trans)
        elif self.activation == 'none':
            return lin_trans
        else:
            raise RuntimeError("Unsupported activation function.")
        
    def update_node():
        pass

