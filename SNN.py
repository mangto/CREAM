import numpy, random

import tool.Csys as Csys # -> for system control
import visualizer
from Functions import * # -> afunctions
import tool.datasets as dataset


class snn:
    input_type = [list, numpy.array, numpy.ndarray]

    # 
    # SNN is an abbreviation of Simple Neural Network.
    # It has only one hidden layer
    # This is made to vertify if delta rule works
    # 
    # If you're reading this, you're on my github 
    # if not, please contact mangto0701@gmail.com
    # 

    def init_weights(NetworkShape:list):
        result = [
            [
                list(numpy.random.randn(NetworkShape[i+1])*0.1) for j in range(n)
            ] for i, n in enumerate(NetworkShape[:-1])
        ]

        return result

    def init_biases(NetworkShape:list):
        result = [
            list(numpy.zeros(i)) for i in NetworkShape[1:]
        ]

        return result

    def reset_activation(self):
        return numpy.array([numpy.zeros(hn) for hn in self.NetworkShape],dtype=object)    

    def __init__(self, NetworkShape:list, function:sigmoid, LearningRate:float=0.3):

        if (len(NetworkShape) != 3): raise ValueError("That's not 'snn' a.k.a. Simple Neural Network")

        self.NetworkShape = NetworkShape
        self.function = function
        self.l_rate = LearningRate

        self.layercount = len(NetworkShape)
        self.weights = snn.init_weights(NetworkShape)
        self.biases = snn.init_biases(NetworkShape)

        self.activations = self.reset_activation()
        self.derive_activations = self.reset_activation()

        self.actives = self.reset_activation()
        self.dactivations = self.reset_activation()


    def forward(self, input:list):
        if (type(input) not in snn.input_type): raise ValueError(f"To do forward, input type have to be list or numpy.array, not {type(input)}")
        if (len(input) != self.NetworkShape[0]): raise ValueError(f"Wrong input counts, need: {self.NetworkShape[0]} taken: {len(input)}")

        self.activations = self.reset_activation()
        self.pure_activations = self.reset_activation()
        
        self.activations[0] = input
        self.derive_activations[0] = input

        for i in range(self.layercount-1):
            new_activ = MultiplyEach(self.weights[i],self.activations[i])
            new_activ = numpy.sum(new_activ,axis=0)+self.biases[i]
            self.derive_activations[i+1] = [self.function(i, Derivative=True) for i in new_activ]
            self.activations[i+1] =[self.function(i) for i in new_activ]

    def backpropgation(self, target:list): # target is a list of numbers that our network have to make out
        
        if (type(target) not in snn.input_type): raise ValueError(f"To do forward, target type have to be list or numpy.array, not {type(target)}")
        if (len(target) != self.NetworkShape[-1]): raise ValueError(f"Wrong target counts, need: {self.NetworkShape[0]} taken: {len(target)}")

        #
        delta = numpy.array(self.activations[-1]) - target
        self.weights[1] -= self.l_rate * numpy.transpose(numpy.reshape(delta, (self.NetworkShape[-1], 1)) * self.activations[1])
        self.biases[1] -= self.l_rate * delta

        h = numpy.array(self.activations[1])
        delta = numpy.sum(self.weights[1] * delta,axis=1)*h*(1-h)
        self.weights[0] -= self.l_rate * numpy.transpose(numpy.reshape(delta, (self.NetworkShape[1], 1)) * self.activations[0])
        self.biases[0] -= self.l_rate * delta

    def save_weight(self):
        return [list(w) for w in self.weights]
        
    def save_bias(self):
        return [list(b) for b in self.biases]

    def load_weight(self, weight):
        self.weights = [numpy.array(w) for w in weight]

    def load_bias(self, bias):
        self.biases = [numpy.array(b) for b in bias]