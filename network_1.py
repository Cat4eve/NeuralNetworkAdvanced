import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases  = np.zeros(output_size)
        self.activation : str = activation
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
        return self.output
    
    def backward(self, grad_output, learning_rate):
        grad_weights = np.dot(self.inputs.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0)
        
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input
    
class DenseNetwork:
    def __init__(self):
        self.layers : list[DenseLayer] = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad_output, learning_rate):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, learning_rate)