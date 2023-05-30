import numpy as np
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from network_1 import DenseNetwork, DenseLayer

# Data generation

X, y = make_regression(n_samples=100, n_features=10, noise=0.5, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Neural network implementation test

learning_rate = 0.001
num_epochs = 1000

nn = DenseNetwork(num_epoches=num_epochs, alpha=learning_rate)
nn.add_layer(DenseLayer(10,3))
nn.add_layer(DenseLayer(3,1))
nn.call(X = X_train, y = y_train)

print("Mean Squared Error (DenseNetwork implemented from scratch):", r2_score(y_test, nn.call(X = X_test, training=False)))



