import tensorflow as tf
import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = tf.Variable(tf.random.normal([input_size, output_size], stddev=0.01, dtype=tf.double))
        self.biases = tf.Variable(tf.zeros([output_size], dtype=tf.double))
        self.activation = activation
    
    def forward(self, inputs):
        self.inputs = inputs
        self.output = tf.matmul(inputs, self.weights) + self.biases
        
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.output)
        elif self.activation == 'sigmoid':
            self.output = tf.nn.sigmoid(self.output)
        
        return self.output

class DenseNetwork:
    def __init__(self, num_epochs=1000, alpha=0.001, beta_1=0.9, beta_2=0.97):
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.betas = np.array([beta_1, beta_2])
        self.layers = []
    
    def add_layer(self, layer):
        self.layers.append(layer)
    
    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs
    
    def trainable_variables(self):
        variables = []
        for layer in self.layers:
            variables.extend([layer.weights, layer.biases])
        return variables
    
    def call(self, X, y=None, training=True):
        if not training:
            return self.forward(X)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.alpha, beta_1=self.betas[0], beta_2=self.betas[1])

        for _ in range(self.num_epochs):
            with tf.GradientTape() as tape:
                pred = self.forward(X)
                loss = tf.reduce_mean(tf.square(pred - y))

            gradients = tape.gradient(loss, self.trainable_variables())
            optimizer.apply_gradients(zip(gradients, self.trainable_variables()))

        return self.call(X, y, False)

# test
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nn = DenseNetwork(num_epochs=300)
nn.add_layer(DenseLayer(10,1,activation='sigmoid'))
nn.call(X = X_train, y = y_train)

print("Accuracy (DenseNetwork implemented from scratch):", accuracy_score(y_test, [0 if i < 0.5 else 1 for i in nn.call(X = X_test, training=False)]))
