import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None, eps=1e-8):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases  = np.zeros(output_size)
        self.activation = activation
        self.eps = eps
        self.m_dw = np.zeros_like(self.weights)
        self.v_dw = np.zeros_like(self.weights)
        self.m_db = np.zeros_like(self.biases)
        self.v_db = np.zeros_like(self.biases)
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = np.maximum(0, self.output)
        elif self.activation == 'sigmoid':
            self.output = 1 / (1 + np.exp(-self.output))
        
        return self.output
    
    def backward(self, grad_output, learning_rate, beta1=0.9, beta2=0.97):
        m_dw = self.m_dw
        v_dw = self.v_dw
        m_db = self.m_db
        v_db = self.v_db
        
        num_samples = self.inputs.shape[0]
        
        grad_weights = np.dot(self.inputs.T, grad_output) / num_samples
        grad_biases = np.sum(grad_output, axis=0) / num_samples
        grad_input = np.dot(grad_output, self.weights.T)
        
        if self.activation == 'relu':
            grad_input[self.inputs <= 0] = 0
        elif self.activation == 'sigmoid':
            e_x = 1 / (1 + np.exp(-self.inputs))
            grad_input = e_x * (1 - e_x) * grad_input
        
        # m <- first moment
        # v <- second moment
        m_dw = beta1 * m_dw + (1 - beta1) * grad_weights
        v_dw = beta2 * v_dw + (1 - beta2) * np.square(grad_weights)
        m_db = beta1 * m_db + (1 - beta1) * grad_biases
        v_db = beta2 * v_db + (1 - beta2) * np.square(grad_biases)
        
        self.weights -= learning_rate * m_dw / (np.sqrt(v_dw) + self.eps)
        self.biases -= learning_rate * m_db / (np.sqrt(v_db) + self.eps)
        
        return grad_input
    
class DenseNetwork:
    def __init__(self, num_epoches=1000, alpha=0.001, beta1=0.9, beta2=0.97):
        self.num_epoches = num_epoches
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.alpha, self.beta1, self.beta2)

    def call(self, X, y=None, training=True):
        if not training:
            return self.forward(X)

        for _ in range(self.num_epoches):
            pred = self.forward(X)
            gradient = 2 / len(X) * (pred - y)  # with respect to the loss=sum((pred-y)^2)
            self.backward(gradient)

        accuracy = accuracy_score(y, np.round(self.forward(X)))
        return accuracy


# test
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

X, y = make_classification(n_samples=100, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

network = DenseNetwork(num_epoches=100)
network.add_layer(DenseLayer(input_size=10, output_size=10))
network.add_layer(DenseLayer(input_size=10, output_size=1))
network.call(X_train, y_train)

accuracy = network.call(X_test, y_test)
print("Accuracy:", accuracy)
