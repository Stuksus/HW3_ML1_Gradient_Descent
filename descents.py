from dataclasses import dataclass
from enum import auto
from enum import Enum
from typing import Dict
from typing import Type
from numpy import linalg
import numpy as np
import random
import sympy as sy

@dataclass
class LearningRate:
    lambda_: float = 1e-3
    s0: float = 1
    p: float = 0.5

    iteration: int = 0

    def __call__(self):
        """
        Calculate learning rate according to lambda (s0/(s0 + t))^p formula
        """
        self.iteration += 1
        return self.lambda_ * (self.s0 / (self.s0 + self.iteration)) ** self.p


class LossFunction(Enum):
    MSE = auto()
    MAE = auto()
    LogCosh = auto()
    Huber = auto()


class BaseDescent:
    """
    A base class and templates for all functions
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        """
        :param dimension: feature space dimension
        :param lambda_: learning rate parameter
        :param loss_function: optimized loss function
        """
        self.w: np.ndarray = np.random.rand(dimension)
        self.lr: LearningRate = LearningRate(lambda_=lambda_)
        self.loss_function: LossFunction = loss_function

    def step(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.update_weights(self.calc_gradient(x, y))

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        Template for update_weights function
        Update weights with respect to gradient
        :param gradient: gradient
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        pass

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Template for calc_gradient function
        Calculate gradient of loss function with respect to weights
        :param x: features array
        :param y: targets array
        :return: gradient: np.ndarray
        """
        pass

    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        return np.mean((y - (x @ self.w))**2)
#         # TODO: implement loss calculation function
#         raise NotImplementedError('BaseDescent calc_loss function not implemented')

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate predictions for x
        :param x: features array
        :return: prediction: np.ndarray
        """
        return x @ self.w
#         # TODO: implement prediction function
#         raise NotImplementedError('BaseDescent predict function not implemented')


class VanillaGradientDescent(BaseDescent):
    """
    Full gradient descent class
    """

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        lr = self.lr()
        self.w -= gradient * lr
        return -gradient * lr
#         # TODO: implement updating weights function
#         raise NotImplementedError('VanillaGradientDescent update_weights function not implemented')

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        difference = 2*((np.dot(np.dot(x.T,x),self.w) - np.dot(x.T,y)) / y.shape[0])
        return difference
#         # TODO: implement calculating gradient function
#         raise NotImplementedError('VanillaGradientDescent calc_gradient function not implemented')


class StochasticDescent(VanillaGradientDescent):
    """
    Stochastic gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_)
        self.batch_size = batch_size
        self.iteration = 0
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:

        difference = 2*((np.dot(np.dot(x[:self.iteration+self.batch_size].T,x[:self.iteration+self.batch_size]),self.w) - np.dot(x[:self.iteration+self.batch_size].T,y[:self.iteration+self.batch_size])) / y[:self.iteration+self.batch_size].shape[0])
        try:
            self.iteration = random.randint(1,x.shape[0]//100)
        except ValueError:
            self.iteration = 0
        return difference   
#         # TODO: implement calculating gradient function
#         raise NotImplementedError('StochasticDescent calc_gradient function not implemented')


class MomentumDescent(VanillaGradientDescent):
    """
    Momentum gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.h * self.alpha + self.lr() * gradient
        self.w -= self.h
        return - self.h
#         # TODO: implement updating weights function
#         raise NotImplementedError('MomentumDescent update_weights function not implemented')


class Adam(VanillaGradientDescent):
    """
    Adaptive Moment Estimation gradient descent class
    """

    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0

    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.m = self.beta_1*self.m + (1 - self.beta_1)*gradient
        self.v = self.beta_2*self.v + (1 - self.beta_2)*(gradient)**2
        m_hat = self.m /(1-self.beta_1**self.iteration)  
        v_hat = self.v /(1-self.beta_2**self.iteration)
        diff = (self.lr() * m_hat)/(np.sqrt(v_hat)+self.eps)
        self.w -= diff
        return -diff
#         # TODO: implement updating weights function
#         raise NotImplementedError('Adagrad update_weights function not implemented')


class BaseDescentReg(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, mu: float = 0, **kwargs):
        """
        :param mu: regularization coefficient (float)
        """
        super().__init__(*args, **kwargs)

        self.mu = mu

    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate gradient of loss function and L2 regularization with respect to weights
        """
#         l2_gradient: np.ndarray = np.zeros_like(x.shape[1])  # TODO: replace with L2 gradient calculation

        l2_gradient = np.append(self.w[:-1],0).T

        return super().calc_gradient(x, y) + l2_gradient * self.mu

class VanillaGradientDescentReg(BaseDescentReg, VanillaGradientDescent):
    """
    Full gradient descent with regularization class
    """


class StochasticDescentReg(BaseDescentReg, StochasticDescent):
    """
    Stochastic gradient descent with regularization class
    """


class MomentumDescentReg(BaseDescentReg, MomentumDescent):
    """
    Momentum gradient descent with regularization class
    """


class AdamReg(BaseDescentReg, Adam):
    """
    Adaptive gradient algorithm with regularization class
    """
    
    
    
    
    
class BaseDescentLogCosh(BaseDescent):
    """
    A base class with regularization
    """

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
    
    def calc_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate loss for x and y with our weights
        :param x: features array
        :param y: targets array
        :return: loss: float
        """
        pre = np.log1p(np.cosh((x.to_numpy() @ self.w) - y)).to_numpy()
        pre = np.where(pre==np.inf, 1, pre)
        return pre.mean()


    
class VanillaGradientDescentLogCosh(BaseDescentLogCosh):
    """
    Full gradient descent with regularization class
    """ 
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        lr = self.lr()
        self.w -= gradient * lr
        return -gradient * lr
    
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        difference = np.tanh(x.dot(self.w).to_numpy()- y.to_numpy()).dot(x)
        return difference


class StochasticDescentLogCosh(VanillaGradientDescentLogCosh):
    """
    Stochastic gradient descent with regularization class
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3, batch_size: int = 50,
                 loss_function: LossFunction = LossFunction.MSE):
        """
        :param batch_size: batch size (int)
        """
        super().__init__(dimension, lambda_, loss_function)
        self.batch_size = batch_size
        self.iteration = 0
    def calc_gradient(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        difference = np.tanh((x[:self.iteration+self.batch_size] @ self.w).to_numpy() - y[:self.iteration+self.batch_size].to_numpy()).dot(x[:self.iteration+self.batch_size])
        try:
            self.iteration = random.randint(1,x.shape[0]//100)
        except ValueError:
            self.iteration = 0
        return difference


class MomentumDescentLogCosh(VanillaGradientDescentLogCosh):
    """
    Momentum gradient descent with regularization class
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.alpha: float = 0.9

        self.h: np.ndarray = np.zeros(dimension)


    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.h = self.h * self.alpha + self.lr() * gradient
        self.w -= self.h
        return - self.h


class AdamLogCosh(VanillaGradientDescentLogCosh):
    """
    Adaptive gradient algorithm with regularization class
    """
    def __init__(self, dimension: int, lambda_: float = 1e-3, loss_function: LossFunction = LossFunction.MSE):
        super().__init__(dimension, lambda_, loss_function)
        self.eps: float = 1e-8

        self.m: np.ndarray = np.zeros(dimension)
        self.v: np.ndarray = np.zeros(dimension)

        self.beta_1: float = 0.9
        self.beta_2: float = 0.999

        self.iteration: int = 0
            
    def update_weights(self, gradient: np.ndarray) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """
        self.iteration += 1
        self.m = self.beta_1*self.m + (1 - self.beta_1)*gradient
        self.v = self.beta_2*self.v + (1 - self.beta_2)*(gradient)**2
        m_hat = self.m /(1-self.beta_1**self.iteration)  
        v_hat = self.v /(1-self.beta_2**self.iteration)
        diff = (self.lr() * m_hat)/(np.sqrt(v_hat)+self.eps)
        self.w -= diff
        return -diff

def get_descent(descent_config: dict) -> BaseDescent:
    descent_name = descent_config.get('descent_name', 'full')
    regularized = descent_config.get('regularized', False)
    logcosh = descent_config.get('Logcosh', False)

    descent_mapping: Dict[str, Type[BaseDescent]] = {
        'full': VanillaGradientDescentReg if regularized else VanillaGradientDescentLogCosh if logcosh else VanillaGradientDescent ,
        'stochastic': StochasticDescentReg if regularized else StochasticDescentLogCosh if logcosh else StochasticDescent,
        'momentum': MomentumDescentReg if regularized else MomentumDescentLogCosh if logcosh else MomentumDescent,
        'adam': AdamReg if regularized else AdamLogCosh if logcosh else Adam
    }
        
#             descent_mapping: Dict[str, Type[BaseDescent]] = {
#         'full': VanillaGradientDescent if not regularized VanillaGradientDescentLogCosh if not logcosh else VanillaGradientDescentReg,
#         'stochastic': StochasticDescent if not regularized else StochasticDescentReg,
#         'momentum': MomentumDescent if not regularized else MomentumDescentReg,
#         'adam': Adam if not regularized else AdamReg
#     }

    if descent_name not in descent_mapping:
        raise ValueError(f'Incorrect descent name, use one of these: {descent_mapping.keys()}')

    descent_class = descent_mapping[descent_name]

    return descent_class(**descent_config.get('kwargs', {}))
