from layer import Layer
from helpers import data_generator

import pandas as pd
import torch
import matplotlib.pyplot as plt

class Network:
    def __init__(self, structure: list, activation:str):
        self.structure = structure
        # Create layers
        self.layers = [Layer(n_input=structure[i-1], nodes=node, activation=activation, label="Layer "+str(i)) for i, node in enumerate(structure) if i!=0]
        # Edit the final layer attributes.
        self.layers[-1].label = "Output Layer"
        self.layers[-1].activation = "none"
        
    def predict(self, x):
        for layer in self.layers:
            x = layer.f_propogate(x)
        return x
    
    def calc_loss(self, yhat, y, criterion):
        if criterion.lower() == "mse":
            loss = torch.mean((y-yhat)**2)
        else:
            raise RuntimeError("Loss criterion not recognized or not specified. Please correct.")
        return loss
    
    def update_params(self, lr, gamma):
        for layer in self.layers:
            # Calc update step: sum of fraction of previous step and current step.
            update_step_w = gamma*layer.prev_step_w + lr*layer.weights.grad
            update_step_b = gamma*layer.prev_step_b + lr*layer.biases.grad
            # Change previous step tensors to current ones, to be used for next update step.
            layer.prev_step_w = update_step_w
            layer.prev_step_b = update_step_b
            # Update the parameters
            layer.weights = (layer.weights.detach() - update_step_w).requires_grad_()
            layer.biases = (layer.biases.detach() - update_step_b).requires_grad_()
    
    def train(self, data, batch_size, epochs, lr, gamma):
        self.losses = []
        self.test_losses = []
        for epoch in range(epochs):
            # Prepare data: train-test split, standardize and batch
            train_data_gen, testing_data = data_generator(data, batch_size)
            X_test, y_test = testing_data
            # train over a number of iterations and keep track of the losses.
            losses_iters = []
            for features, targets in train_data_gen:
                # Make predictions on the data
                yhat = self.predict(features)
                # Calculate the loss
                loss = self.calc_loss(yhat=yhat, y=targets, criterion="mse")
                # Perform backpropogation
                loss.backward()
                # Update parameters
                self.update_params(lr, gamma)
                    
                losses_iters.append(loss.detach().item())
            # Calculate average loss of this epoch and append to list.
            self.losses.append(sum(losses_iters)/len(losses_iters))
            # Calculate loss of testing set
            yhat_test = self.predict(X_test)
            loss_test = self.calc_loss(yhat_test, y_test, 'mse')
            self.test_losses.append(loss_test.detach().item())
                     
if __name__ == "__main__":
    # Create model
    model = Network([8,10,1], 'tanh')
    # Modify the activation function of the output layer.
    model.layers[-1].activation = 'relu'
    # Collect and prepare the data
    data = pd.read_csv('concrete_data.csv')
    model.train(data=data, batch_size=8, epochs=100, lr=0.001, gamma=0.001)

    # Plot y1 and y2 against x, show the legend

    # Plot the losses over time
    iterations = [i+1 for i in range(len(model.losses))]
    plt.plot(iterations, model.losses, label="Training loss")
    plt.plot(iterations, model.test_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Average loss")
    plt.legend()
    # plt.plot(iterations, model.d_losses)
    plt.show()   
    