import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from network import Network
from helpers import generate_test_data

# Generate test data of y = ax+b but with artificial gaussian error variance
x, y = generate_test_data(2, 10, 50, 0, .5)
x = np.asarray(x).reshape([50, 1])
y = np.asarray(y).reshape([50, 1])

# Create network and generate predictions
nn = Network(input_layer=1, structure=[10,10,1])
predictions = nn.f_propogate(x)

# Calculate loss using mean squared error function
loss = nn.calc_loss(y_pred=predictions, y_true=y, function='mse')
print(loss)