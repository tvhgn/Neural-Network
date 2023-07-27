from node import Node

import numpy as np

class HiddenLayer:
    def __init__(self, input_array, num_nodes, activation):
        # Empty list for containing nodes and a separate list for their outputs
        self.nodes = []
        self.outputs = []
        # Create nodes
        for _ in range(num_nodes):
            node = Node(input_array.shape[0], activation=activation)
            self.nodes.append(node)
            self.outputs.append(node.compute_output(input_array))
        # Cast to numpy array
        self.outputs = np.asarray(self.outputs)
            
def create_network(input_array, structure):
    network = []
    num_layers = len(structure)
    for i in range(num_layers):
        if i == 0:
            network.append(HiddenLayer(input_array, structure[0], activation='relu'))
        else:
            new_layer = HiddenLayer(network[i-1].outputs, structure[i], activation='relu')
            network.append(new_layer)
    return network

# testing
np.random.seed(23)
input_layer = np.random.uniform(0,10,3)
print(input_layer)     
network = create_network(input_array=input_layer, structure=[5,10,5,1])
print(network[-1].outputs)

    