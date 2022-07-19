import numpy

def sigmoid(value):
    return 1/(1+numpy.exp(-1*value))

def RotateWeight(weights:numpy.array):
    return numpy.flip(numpy.rot90(weights,k=-1),1)

def CostFunctionDerivative(output:list, real:list):
    out = numpy.array(output)
    real = numpy.array(real)

    cost = -2*(out-real)

    return cost

class network:
    def __init__(self, LearningRate:float, NetworkShape:list):
        self.NetworkShape = NetworkShape
        self.weights = [RotateWeight(numpy.array([list(numpy.random.randn(i)) for j in range(NetworkShape[k+1])])) for k, i in enumerate(NetworkShape[0:-1])]
        self.activations = self.reset_activation()

        self.l_rate = LearningRate

    def reset_activation(self):
        self.activations = [ [0 for i in range(hn)] for hn in self.NetworkShape]

    def forward(self,inputs:list):
        if (len(inputs) != self.NetworkShape[0]): raise ValueError("Wrong input counts")
        self.reset_activation()

        self.activations[0] = inputs
        for i in range(len(self.activations[0:-1])):
            act = self.activations[i]
            weights = self.weights[i]
            
            active = sigmoid(numpy.sum(RotateWeight(weights)*act,axis=1))

            self.activations[i+1] = active


    def backpropgation(self, RValue:list):
        if (len(RValue) != self.NetworkShape[-1]): raise ValueError("Worng Counts of 'RValue' (=RValue)")
        
        cost = CostFunctionDerivative(self.activations[-1],RValue)

        for weight in reversed(self.weights):
            pass
        
count = 1

net = network(0.6,[2,3,2])

import random

for i in range(count):
    a = [1,0]
    net.forward(a)
    print(net.activations)
