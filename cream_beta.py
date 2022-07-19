import numpy

def sigmoid(value):
    return 1/(1+numpy.exp(-1*value))

def RotateWeight(weights:numpy.array):
    return numpy.flip(numpy.rot90(weights,k=-1),1)

def CostFunction(output:list, real:list):
    out = numpy.array(output)
    real = numpy.array(real)

    cost = numpy.sum((out-real)**2)

    return cost

class network:
    def __init__(self, LearningRate:float, NetworkShape:list):
        self.NetworkShape = NetworkShape
        self.weights = [numpy.array([list(numpy.random.randn(i)) for j in range(NetworkShape[k+1])]) for k, i in enumerate(NetworkShape[0:-1])]
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
            rotated_weights = RotateWeight(weights)

            active = (sigmoid(numpy.sum(numpy.array([list(a*rotated_weights[j]) for j, a in enumerate(act)]),axis=0)))

            self.activations[i+1] = active


    def backpropgation(self, RValue:list):
        if (len(RValue) != self.NetworkShape[-1]): raise ValueError("Worng Counts of 'RValue' (=RValue)")
        
        cost = CostFunction(self.activations[-1],RValue)
        errors = [RValue - self.activations[-1]]

        for i, weight in enumerate(reversed(self.weights)):
            new_error = numpy.sum(RotateWeight(weight)*errors[i],axis=1)
            errors.append(new_error)

        print(cost)
        print(errors)

count = 1

net = network(0.6,[729,16,16,10])

import random

for i in range(count):
    a = sigmoid(numpy.random.randn(729))
    net.forward(a)
    net.backpropgation([1,0,0,0,0,0,0,0,0,0])

open('.\\dat','w').write(str(net.activations))
