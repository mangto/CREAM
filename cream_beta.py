import numpy, random

import Csys # -> for system control
import visualizer
from Functions import * # -> afunctions
import datasets as dataset


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
        self.actives = self.reset_activation()

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
                costs = costs + CostFunction(self.activations[-1], data[1])
                dcosts = dcosts + CostFunction(self.activations[-1], data[1], True)
                actives = actives + self.activations
            costs /= PackCount
            dcosts /= PackCount
            actives /= PackCount
            self.actives = actives

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
        result = costs*self.actives[l][n]*[self.function(a,Derivative=True) for a in self.actives[-1]]

        for i, weight in enumerate(weights[1:]):
            Csys.out(f'{i} | {weight}', Csys.bcolors.OKCYAN)



        return sum(result)

    def backpropgation(self, DerivativeCost, active:list):
        if (len(DerivativeCost) != self.NetworkShape[-1]): raise ValueError("Wrong input counts")

        for l, layer in enumerate(self.weights):
            for n, neuron in enumerate(layer):
                for w, weight in enumerate(neuron):

                    print(Csys.division(50))
                    Csys.out(f"{l} {n} {w}", Csys.bcolors.FAIL, True)

                    weights = []
                    if (l < self.lenW-2):
                        weights = [self.weights[l+1][w]] + self.weights[l+2:]

                    elif (l == self.lenW-2):
                        weights = [self.weights[l+1][w]]

                    dw = self.PartialDerivative(weights, DerivativeCost, l, n, w)
                    self.weights[l][n][w] -= self.weights[l][n][w]*dw*self.learning_rate