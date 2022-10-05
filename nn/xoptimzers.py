from this import d
import numpy as np

class gradient_descent:
    '''
    Gradient descent
    w := w * lr * gradient array
    b := b * lr * gradient array
    '''

    def __init__(self, learning_rate : float = 0.01) -> None:
        self.learning_rate = learning_rate

    def gradient(self,  weights : np.ndarray,
                        bias : np.ndarray,
                        dW : np.ndarray, 
                        dB : np.ndarray) -> np.ndarray:
        weights = weights -     self.learning_rate * dW
        bias    = bias  -       self.learning_rate * dB
        return weights, bias