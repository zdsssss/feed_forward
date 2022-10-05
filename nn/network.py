import numpy as np

class network:

    def __init__(self) -> None:
        self.layers = []
    
    def fit(self,   x_train : list, 
                    y_train : list,
                    ep : int,
                    optimizer : object,
                    loss_function : object) -> None:
        self.x = x_train
        self.y = y_train
        self.ep = ep
        self.optimizer = optimizer
        self.loss_function = loss_function

        for i in range(ep):
            self.__train()
            if i % 500 == 0:
                print(f"ep {i} coast: {self.coast}")

    def __train(self) -> None:
        costGradientW = []
        costGradientB = []
        self.backward()

    def backward(self) -> None:
        ibb = np.random.randint(0,4)

        batch = self.x[ibb]
        net_out = self.predict(batch)
        y = self.y[ibb]
        self.coast = self.loss_function.loss(net_out, y)

        outputs = [""]
        for layer in self.layers:
            outputs.append(str(layer.backward_propagation(self.coast)))
        return outputs

    def batch(self) -> None:
        raise NotImplementedError

    def add(self, a : object) -> None:
        self.layers.append(a)

    def predict(self, x : np.ndarray) -> np.ndarray:
        outputs = x
        for layer in self.layers:
            outputs = layer.forward_propagation(outputs)
        return outputs








    def predict_debug(self, x : np.ndarray) -> np.ndarray:
        outputs = x
        for layer in self.layers:
            outputs = layer.forward_propagation(outputs)
            print("weights : ",layer.weights)
            print("outs : ",outputs)
            print("#"*100)
        return outputs
    
    def debug(self) -> object:
        return self.layers