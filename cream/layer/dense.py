from cream.functions import *
import numpy


class Dense:
    def __init__(self, size:int, activation=None, InputShape:int=None):
        '''
        normal neural network layer
        '''

        self.size : int = size
        self.activation = activation if activation else ReLU
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

        self.weights = numpy.random.uniform(-1.0, 1.0, (self.size, previous_shape)) * 0.01
        self.biases = numpy.zeros((self.size))

    def do(self, previous_activation:list | numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        '''
        forward feed of dense layer
        '''

        self.raw_activ = numpy.sum(numpy.array(self.weights) * numpy.array(previous_activation), axis=1) + self.biases
        self.activ = self.activation(self.raw_activ)
        return self.raw_activ, self.activ

    def get_weights(self):
        return self.weights

    def backpropagation(self, args:dict) -> list | numpy.ndarray:
        delta :numpy.ndarray = args.get('delta')
        layers :list = args.get('layer')
        index :int = args.get('index')
        lrate :int = args.get('lrate')
        activation :list = args.get('activation')

        if (index == -1):
            # delta = delta * self.activation(self.raw_activ, True)
            self.weights -= lrate * numpy.outer(delta, activation[index - 1])
            self.biases -= lrate * delta

            return delta
            
        else:
            delta = delta * numpy.transpose(layers[index + 1].get_weights()) * self.activation(self.raw_activ, True)[:,None]
            delta = numpy.sum(delta, axis=1) / len(delta[0])
            self.weights -= lrate * numpy.outer(delta, activation[index - 1])
            self.biases -= lrate * delta

            return delta