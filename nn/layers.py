import numpy as np
class layer_dense:

    def __init__(self, n_inputs : int, n_neurons : int, activation : object = None) -> None:
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation_f = activation
 
    def forward_propagation(self, inputs : np.ndarray) -> np.ndarray:
        self.inputs = inputs
        if self.activation_f:
            self.a = np.dot(self.inputs, self.weights) + self.biases
            self.outputs = self.activation_f(self.a) 
        else:
            self.a = np.dot(self.inputs, self.weights) + self.biases
            self.outputs = self.a
        return self.outputs
  
    def backward_propagation(self, error_ : np.ndarray) -> list[np.ndarray]:
        print(error_)
        print(self.outputs)
        -(error_/self.outputs)
        self.wde = (-(error_/self.outputs)+\
                    ((1-error_)/(1-self.outputs))) *\
                    self.activation_f.derivative(self.a) * \
                    self.inputs
        print(self.wde)
        self.bde = error_ * self.activation_f.derivative
        return self.wde, self.bde