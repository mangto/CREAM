#
# Code of Cream Beta


import numpy
import Csys

def sigmoid(value):
    return 1/(1+numpy.exp(-1*value))

def RotateWeight(weights:numpy.array):
    return numpy.flip(numpy.rot90(weights,k=-1),1)

def MultiplyEach(a:numpy.array,b:numpy.array):
    if (len(a) != len(b)): raise ValueError(f"Different inputs {len(a)}, {len(b)}")
    return numpy.array([list(a[i]*b[i]) for i in range(len(a))])

def Multiply(a:numpy.array,b:numpy.array):
    if (len(a[0]) != len(b)): raise ValueError(f"Different inputs ({len(a[0])}*{len(a)}), {len(b)}")
    return numpy.array([list(a[i]*b) for i in range(len(a))])

def CostFunctionDerivative(output:list, real:list):
    out = numpy.array(output)
    real = numpy.array(real)

    cost = (out-real)

    return cost

def CostFunction(output:list, real:list):
    out = numpy.array(output)
    real = numpy.array(real)

    cost = (out-real)**2/2

    return cost

def NeuronCounts(shape:list):
    out = 0
    for i in range(len(shape)-1):
        out += shape[i]*shape[i+1]

    return out

class network:
    def __init__(self, LearningRate:float, NetworkShape:list):
        self.NetworkShape = NetworkShape
        self.weights = [RotateWeight(numpy.array([list(numpy.random.randn(i)) for j in range(NetworkShape[k+1])])) for k, i in enumerate(NetworkShape[0:-1])]
        self.activations = self.reset_activation()

        self.l_rate = LearningRate
        self.NueronCount = NeuronCounts(NetworkShape)

    def reset_activation(self):
        self.activations = [ [0 for i in range(hn)] for hn in self.NetworkShape]

    def forward(self,inputs:list):
        if (len(inputs) != self.NetworkShape[0]): raise ValueError("Wrong input counts")
        self.reset_activation()

        self.activations[0] = inputs
        for i in range(len(self.activations[0:-1])):
            act = self.activations[i]
            weights = self.weights[i]

            try:
                active = sigmoid((numpy.sum(RotateWeight(weights)*act,axis=1)))
            except:
                Csys.stop(f"Pause because of error\nweights: \n{weights}\n\nactivation: \n{act}")

            self.activations[i+1] = active

    def PartialDerivative(self,weights:list, costs, SelfActivation):
        result = 1 #assume cost is 1 -> multiply later
            
        if (len(weights) > 1):               
            result = weights[0]
            reverse = list(reversed(weights[1:-1]))
            last = RotateWeight(weights[-1])
            out = []

            for i in range(self.NetworkShape[-1]):
                c = last[i]
                    

                for weight in reverse:
                    c = numpy.sum(Multiply(weight, c),axis=1)

                out.append(c)
            result = numpy.sum([numpy.sum(result*o,axis=0) for o in out]*costs)
        output = SelfActivation*result
        return output

    def backpropgation(self, RValue:list):
        if (len(RValue) != self.NetworkShape[-1]): raise ValueError("Worng Counts of 'RValue' (=RValue)")

        cost = CostFunctionDerivative(self.activations[-1],RValue)
        lenW = len(self.weights)

        for l, layer in enumerate(self.weights):
            for n, neuron in enumerate(layer):
                for w, weight in enumerate(neuron):
                    active = self.activations[l][n]

                    try:
                        if (l >= lenW-2):
                            weights = []
                        else:
                            weights = [self.weights[l+1][w]] + list(self.weights[l+2:])
                    except Exception as e:
                        Csys.stop(f"Pause because of error\n{e}\nlen: {len(self.weights)} | l: {l}")
                    dw = self.PartialDerivative(weights, cost, active)
                    self.weights[l][n][w] = self.weights[l][n][w] - self.weights[l][n][w]*dw*self.l_rate

import random

count = 1000

net = network(0.3,[2,1])
3
dataset = [
    [[0,0], [0]],
    [[0,1],[1]],
    [[1,0],[1]],
    [[1,1],[0]]
]

for i in range(count):
    a = random.sample(dataset,1)[0]
    net.forward(a[0])
    net.backpropgation(a[1])

    cost = numpy.sum(CostFunction(net.activations[-1],a[1]))
    print(f"{i} | {cost}")

Csys.stop()
