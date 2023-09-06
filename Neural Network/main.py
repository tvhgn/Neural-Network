import numpy as np
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler

from network import Network
from helpers import generate_test_data

# Generate test data of y = ax+b but with artificial gaussian error variance
x, y = generate_test_data(2, 10, 100, 0, .5)
x = np.asarray(x).reshape([10, 10])
y = np.asarray(y).reshape([10, 10])

# Create network and generate predictions
nn = Network(input_layer=10, structure=[3,3,1])
predictions = nn.f_propogate(x)

# Calculate loss using mean squared error function
loss = nn.calc_loss(y_pred=predictions, y_true=y, function='mse')
print(loss)