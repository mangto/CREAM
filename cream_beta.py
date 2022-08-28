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

        result = [[numpy.random.randn(NetworkShape[i])* 0.1 for j in range(shape)]
                    for i, shape in enumerate(NetworkShape[1:])]

        return result

    def init_biases(NetworkShape:list):
        # initialize biases with network shape

        result = [numpy.random.randn(shape)*0.1 for shape in NetworkShape[1:]]

        return result

    def reset_activation(NetworkShape):
        # reset activations

        result = [numpy.zeros(shape) for shape in NetworkShape]

        return result


    def __init__(self, NetworkShape:list, ActivationFunction=sigmoid, LearningRate:float=1,
                    weights:numpy.array=None, biases:numpy.array=None):
        self.NetworkShape = NetworkShape
        self.acfunc = ActivationFunction
        self.lrate = LearningRate

        self.weights = weights if weights else network.init_weights(NetworkShape)
        self.biases = biases if biases else network.init_biases(NetworkShape)

        self.activ = network.reset_activation(NetworkShape)
        self.raw_activ = network.reset_activation(NetworkShape)

        self.depth = len(NetworkShape)

    
    def forward(self, input:list):
        assert type(input) in network.InputType, "Wrong Type of Input"
        assert len(input) == self.NetworkShape[0], f"Wrong Count of Input, need: {self.NetworkShape[0]} taken: {len(input)}"

        self.activ = network.reset_activation(self.NetworkShape)
        self.raw_activ = network.reset_activation(self.NetworkShape)
        self.activ[0] = input
        self.raw_activ[0] = input

        for i in range(len(self.NetworkShape[1:])):
            raw = numpy.sum(numpy.array(self.weights[i]) * numpy.array(self.activ[i]), axis=1) + self.biases[i]

            self.raw_activ[i+1] = raw
            self.activ[i+1] = self.acfunc(raw)

    def backpropgation(self, target:list, activations=None):
        assert type(target) in network.InputType, "Wrong Type of Target"
        assert len(target) == self.NetworkShape[-1], f"Wrong Count of Input, need: {self.NetworkShape[-1]} taken: {len(target)}"

        activations = activations if activations else self.activ

        error = activations[-1] - target
        delta = error
        dws = []

        for l, layer in enumerate(reversed(self.weights)):
            
            l = self.depth-2-l # reverse sequence


            if (l < self.depth-2): # if layer is not hidden_last to output
                delta = numpy.sum(delta * numpy.transpose(self.weights[l+1]), axis=1) * self.acfunc(self.raw_activ[l+1],True)
            
            else:
                delta = delta * self.acfunc(self.raw_activ[l+1], True)

            dw = numpy.dot(delta[:,None], numpy.array(self.activ[l])[None]) * self.lrate
            dws.append(dw)


            self.weights[l] -= self.weights[l] * dw * self.activ[l]
            self.biases[l] -= numpy.sum(dw, axis=1) * self.biases[l]