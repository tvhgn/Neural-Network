from node import Node

import numpy as np

class NodeIssue(Exception):
    pass

class Layer:
    def __init__(self, label, input_size, num_nodes, activation):
        # Empty list for containing nodes and a separate list for their outputs
        self.label = "Layer "+label
        self.nodes = []
        self.outputs = []
        
        # Create nodes
        for _ in range(num_nodes):
            node = Node(input_size, activation=activation)
            self.nodes.append(node)
            # self.outputs.append(node.compute_output(input_array))
        # # Cast to numpy array
        # self.outputs = np.asarray(self.outputs)
    
    def __repr__(self) -> str:
        return f"Layer with {len(self.nodes)} nodes"
        
    def f_propogate(self, input_array):
        self.outputs = []
        for i in range(len(self.nodes)):
            self.outputs.append(self.nodes[i].compute_output(input_array))
        return np.asarray(self.outputs)
        
    