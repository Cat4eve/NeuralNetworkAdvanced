import tensorflow as tf

class NeuralLayer(tf.Module):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.w = tf.Variable(
            initial_value=tf.random.normal(
                shape=(self.input_size, self.output_size),
                dtype=tf.float64
            ),
            trainable=True,
        )

        self.b = tf.Variable(
            initial_value=tf.zeros(output_size, dtype=tf.float64),
            trainable=True
        )

    def forward(self, inputs):
        self.inputs = inputs
        self.output = tf.matmul(inputs, self.w) + self.b
        
        if self.activation == 'relu':
            self.output = tf.nn.relu(self.output)
        elif self.activation == 'sigmoid':
            self.output = tf.sigmoid(self.output)
        
        return self.output


class NeuralNetwork(tf.Module):
    def __init__(self, num_epoch, alpha=0.001):
        super().__init__()
        self.num_epoch = num_epoch
        self.alpha = alpha
        self.layers = []

    def add_layers(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def __call__(self, X, y=None, training=True):
        if not training:
            return self.forward(X)

        for _ in range(self.num_epoch):
            with tf.GradientTape(persistent=True) as tape:
                loss = self.forward(X)
            
            variables = [layer.w for layer in self.layers]
            gradients = tape.gradient(loss, variables)

            del tape
        
        return self.__call__(X, y, False)


# test
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score

X, y = make_classification(n_samples=100, n_features=10, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train = tf.cast(X_train, dtype=tf.float64)
X_test = tf.cast(X_test, dtype=tf.float64)
y_train = tf.cast(y_train, dtype=tf.float64)

neural_network = NeuralNetwork(num_epoch=100, alpha=0.001)
neural_network.add_layers(NeuralLayer(input_size=10, output_size=10, activation='relu'))
neural_network.add_layers(NeuralLayer(input_size=10, output_size=1, activation='relu'))
neural_network(X_train, y_train)
predictions = neural_network(X_test, training=False)

binary_predictions = tf.where(predictions >= 0.5, 1, 0)
accuracy = tf.reduce_mean(tf.cast(tf.equal(binary_predictions, y_test), tf.float64))
print("Accuracy:", accuracy.numpy())
