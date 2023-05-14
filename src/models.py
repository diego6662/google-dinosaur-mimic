class Sequential:
    def __init__(self, layers=[]):
        self.layers = layers

    def add(self, layer):
        self.layers.append(layer)

    def run(self, X):
        value = X.copy()
        input_shape = X.shape
        for layer in self.layers:
            layer.set_input_shape(input_shape)
            layer.set_weights()
            layer.set_bias()
            value = layer.compute(value)
            input_shape = value.shape
        return value

