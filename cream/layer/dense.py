from cream.functions import *
import numpy


class Dense:
    def __init__(self, size:int, activation=None, InputShape:int=None):
        '''
        normal neural network layer
        '''

        self.size : int = size
        self.activation = activation if activation else Linear
        self.InputShape = InputShape

        self.weights:numpy.ndarray
        self.biases:numpy.ndarray

        self.activ = numpy.zeros((self.size))
        self.raw_activ = numpy.zeros((self.size))

        # self.backpropagation = True

        return
    
    def __call__(self, previous_activation:list | numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        return self.do(previous_activation)
    
    def reset(self) -> None:
        '''
        reset activation ( + raw_activation )
        '''
        
        self.activ = numpy.zeros((self.size))
        self.raw_activ = numpy.zeros((self.size))

    def generate(self, previous_shape:int | tuple) -> None:
        '''
        generate weights and biases with numpy.random.normal
        '''

        self.weights = numpy.random.normal(size=(self.size, previous_shape)) * 0.1
        self.biases = numpy.zeros((self.size))

    def do(self, previous_activation:list | numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''
        forward feed of dense layer
        '''

        raw = numpy.sum(numpy.array(self.weights) * numpy.array(previous_activation), axis=1) + self.biases
        refined = self.activation(raw)
        return raw, refined

    def backpropagation(self, delta:list | numpy.ndarray) -> list | numpy.ndarray:

        return