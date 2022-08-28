import numpy, random

import tool.Csys as Csys # -> for system control
import visualizer
from Functions import * # -> afunctions
import tool.datasets as dataset

import numpy, random

class network:

    InputType = [list, numpy.array, numpy.ndarray]

    def __str__(self):
        result = f'''
        | type: Cream Neural Network
        | Network Type: Normal
        | Network Shape: {self.NetworkShape}
        | Activation Function: {self.acfunc}
        | Learning Rate: {self.lrate}
        '''

        return result

    def init_weights(NetworkShape:list):
        # initialize weights with network shape

        result = [[numpy.random.randn(NetworkShape[i]) for j in range(shape)]
                    for i, shape in enumerate(NetworkShape[1:])]

        return result

    def init_biases(NetworkShape:list):
        # initialize biases with network shape

        result = [numpy.zeros(shape) for shape in NetworkShape[1:]]

        return result

    def reset_activation(NetworkShape):
        # reset activations

        result = [numpy.zeros(shape) for shape in NetworkShape]


    def __init__(self, NetworkShape:list, ActivationFunction=sigmoid, LearningRate:float=0.3,
                    weights:numpy.array=None, biases:numpy.array=None):
        self.NetworkShape = NetworkShape
        self.acfunc = ActivationFunction
        self.lrate = LearningRate

        self.weights = weights if weights else network.init_weights(NetworkShape)
        self.biases = biases if biases else network.init_biases(NetworkShape)

        self.activations = network.reset_activation(NetworkShape)

    
    def forward(self, input):
        assert type(input) not in network.InputType, "Wrong Type of Input"
        assert len(input) != self.NetworkShape[0], f"Wrong Count of Input, need: {self.NetworkShape[0]} taken: {len(input)}"

        self.activations = network.reset_activation(self.NetworkShape)
