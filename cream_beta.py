import numpy, random

import Csys # -> for system control
from activation_functions import * # -> activation functions
import datasets as dataset

def CostFunction(output, target, dervative:bool=False):
    if (dervative == False): return (numpy.array(output)-numpy.array(target))**2/2
    else: return numpy.array(output)-numpy.array(target)

def MultiplyEach(a:numpy.array,b:numpy.array): # MultiplyEach( [[1, 2], [3, 4]],   [1, 2] ) -> [[1, 2], [6, 8]]
    if (len(a) != len(b)): raise ValueError(f"Different inputs {len(a)}, {len(b)}")
    return numpy.array([a[i]*numpy.array(b[i]) for i in range(len(a))])


class Network:
    learning_rate = 0.3

    def init_neurons(NetworkShape:list):
        net = [[numpy.random.randn(NetworkShape[i+1]) for j in range(NetworkShape[i])] for i in range(len(NetworkShape[:-1]))]
        return net

    def init_bias(NetworkShape:list):
        biases = [numpy.random.randn(1)[0] for l in NetworkShape]
        return biases 

    def __init__(self, NetworkShape:list, learning_rate:float=None, function=ReLU):
        if (learning_rate != None): Network.learning_rate = learning_rate
        
        self.NetworkShape = NetworkShape
        self.function = function

        self.weights = Network.init_neurons(NetworkShape)
        self.lenW = len(self.weights)
        self.bias = numpy.random.randn(1)[0]
        self.activations = self.reset_activation()
        self.real_activations = self.reset_activation()
        self.costs = numpy.zeros(NetworkShape[-1])

    def reset_activation(self):
        return numpy.array([numpy.zeros(hn) for hn in self.NetworkShape],dtype=object)     

    def train(self, PackCount:int, dataset:list, MaxEpoch:int=None, EndCost:float=0.01):
        cost = 0
        epoch = 0

        while (cost > EndCost or epoch == 0 or (MaxEpoch != None and epoch < MaxEpoch)):
            epoch += 1
            costs = numpy.zeros(self.NetworkShape[-1])
            dcosts = numpy.zeros(self.NetworkShape[-1])
            actives = self.reset_activation()

            for i in range(PackCount):
                data = random.sample(dataset, 1)[0]
                self.forward(data[0])
                costs += CostFunction(self.activations[-1], data[1])
                dcosts += CostFunction(self.activations[-1], data[1], True)
                actives = actives + self.activations

            print(actives)
            costs /= PackCount
            actives /= PackCount

            self.backpropgation(dcosts, actives)

    def forward(self, inputs:list):
        if (len(inputs) != self.NetworkShape[0]): raise ValueError("Wrong input counts")
        self.activations = self.reset_activation()
        self.real_activations = self.reset_activation()

        self.activations[0] = inputs
        for l, neurons in enumerate(self.weights):
            activation = self.activations[l]
            new_activation = numpy.sum(MultiplyEach(neurons, activation), axis=0)+self.bias
            self.activations[l+1] = [self.function(a) for a in new_activation]
            self.real_activations[l+1] = new_activation


    def PartialDerivative(self,weights:list, costs, l, n, w):
        result = costs*self.activations[l][n]*[self.function(a,Derivative=True) for a in self.activations[-1]]

        return 0

    def backpropgation(self, DerivativeCost, active:list):
        if (len(DerivativeCost) != self.NetworkShape[-1]): raise ValueError("Wrong input counts")

        for l, layer in enumerate(self.weights):
            for n, neuron in enumerate(layer):
                for w, weight in enumerate(neuron):
                    activation = active[l][n]

                    print(Csys.division(30))
                    print(l, n, w)

                    weights = []
                    if (l < self.lenW-2):
                        weights = [self.weights[l+1][w] + list(self.weights[l+2:])]

                    elif (l == self.lenW-2):
                        weights = [self.weights[l+1][w]]

                    dw = self.PartialDerivative(weights, DerivativeCost, l, n, w)
