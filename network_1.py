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
        
        match self.activation:
            case 'relu':
                if grad_input < 0: grad_input = 0
            case 'sigmoid':
                grad_input = 1/(1+np.exp(-grad_input))*(1-(1/(1+np.exp(-grad_input))))



        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases
        
        return grad_input
    
class DenseNetwork:
    def __init__(self, num_epoches=1000, alpha=0.001, betta_1=.9, betta_2=.97):
        self.num_epoches = num_epoches
        self.alpha = alpha
        self.bettas = np.array([betta_1, betta_2])
        self.layers : list[DenseLayer] = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output, self.alpha)
    
    def call(self, X, y=None, training=True):
        if not training: return self.forward(X)

        for _ in range(self.num_epoches):
            pred = self.forward(X)
            gradient = 2  / len(X) * (pred - y) # with respect to the loss=sum((pred-y)^2)
            self.backward(gradient)

        return self.call(X, y, False)

