import numpy as np
import math

class sigmoid:
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + math.e**(-x))
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        return sigmoid(x)*(1-sigmoid(x))

class tanh:
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return np.tanh(x)
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        return 1.0 - x**2

class relu:
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return np.maximum(0.0, x)
    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        # return 1 if x > 0 else 0
        return np.where(x > 0, 1, 0)

class leaky_relu:
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return np.where(x > 0, x, x * 0.01)
    @staticmethod
    def  derivative(x : np.ndarray) -> np.ndarray:
        return np.where(x < 0, 0.01, 1)

class softmax:
    def __call__(self, x : np.ndarray) -> np.ndarray:
        return np.exp(x) / sum(np.exp(x))

    @staticmethod
    def derivative(x : np.ndarray) -> np.ndarray:
        return 0.01 if x < 0 else 1