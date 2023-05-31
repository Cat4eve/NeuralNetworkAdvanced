import tensorflow as tf

class NeuralLayer(tf.keras.Model):
    def __init__(self, input_size, output_size, activation=None):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.w = tf.Variable(
            initial_value=tf.random_normal_initializer()(
                shape=(self.input_size, self.output_size),
                dtype="float32"
            ),
            trainable=True,
        )

        self.b = tf.Variable(
            initial_value=tf.zeros(output_size),
            trainable=True
        )

    def forward(self, inputs):
        self.inputs = inputs
        self.output = tf.matmul(inputs, self.w) + self.b
        
        match self.activation:
            case 'relu':
                if self.output < 0: self.output = 0
            case 'sigmoid':
                self.output = 1/(1+tf.exp(-self.output))
        
        return self.output



class NeuralNetwork:
    def __init__(self, num_epoch, alpha=0.001):
        self.num_epoch = num_epoch
        self.alpha = alpha
        self.layers : list[NeuralLayer] = []

    def add_layers(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def __call__(self, X, y=None, training=True):
        # super().__call__()
    
        if not training: pass

        for _ in range(self.num_epoch):
            with tf.GradientTape(persistent=True) as tape:
                loss = self.forward(X)
            gradients = tape.gradient(loss, [tf.transpose(i.w) for i in self.layers])

            del tape

        return self.__call__(X, y, False)
    

