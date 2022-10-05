import numpy as np

class SSE:

    @staticmethod
    def loss(y_hat : np.ndarray, y : np.ndarray) -> float:
        return np.sum(np.square( y_hat - y ))

class Mse:

    @staticmethod
    def loss(y_hat : np.ndarray, y : np.ndarray) -> float:
        '''SSE 0.5 '''
        return SSE.loss( y_hat, y ) * 0.5
    
    @staticmethod
    def deriv(y_hat : np.ndarray, y : np.ndarray) -> float:
        '''y hat - y'''
        return y_hat - y

class Binary_crossentropy:
    
    @staticmethod
    def loss(y_pred : np.ndarray, y_true : np.ndarray) -> float:
        # '''SSE * 0.5'''
        # return (np.sum(y*np.log(y_hat)+(1-y)*np.log(1-y_hat))) / len(y) * -1
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        term_0 = (1-y_true) * np.log(1-y_pred + 1e-7)
        term_1 = y_true * np.log(y_pred + 1e-7)
        return -np.mean(term_0+term_1, axis=0)
            
    @staticmethod
    def deriv(y_hat : np.ndarray, y : np.ndarray) -> float:
        '''SSE * 0.5'''
        return SSE.loss( y_hat, y )