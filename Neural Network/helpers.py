import numpy as np
import matplotlib.pyplot as plt

def generate_test_data(a: float, b: float, num_samples: int, mean: float, std_dev: float):
    x = np.random.uniform(0, 1, num_samples)
    errors = np.random.normal(mean, std_dev, num_samples)
    y = a*x + b + errors
    return x, y