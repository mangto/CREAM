import numpy

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

    cost = 2*(out-real)

    return cost

def CostFunction(output:list, real:list):
    out = numpy.array(output)
    real = numpy.array(real)

    cost = (out-real)**2

    return cost

def NeuronCounts(shape:list):
    out = 0
    for i in range(len(shape)-1):
        out += shape[i]*shape[i+1]

    return out

class network:
    def __init__(self, LearningRate:float, NetworkShape:list, round=6):
        self.NetworkShape = NetworkShape
        self.weights = [RotateWeight(numpy.array([list(numpy.random.randn(i)) for j in range(NetworkShape[k+1])])) for k, i in enumerate(NetworkShape[0:-1])]
        self.activations = self.reset_activation()

        self.l_rate = LearningRate
        self.NueronCount = NeuronCounts(NetworkShape)
        self.round = round

    def reset_activation(self):
        self.activations = [ [0 for i in range(hn)] for hn in self.NetworkShape]

    def PartialDerivative(self,weights:list, costs, SelfActivation):
        output = 1
        if (len(weights)>=1):
            result = weights[0] #assume cost is 1 -> multiply later
            
            if (len(weights) >= 2):               
                
                reverse = list(reversed(weights[1:-1]))
                last = RotateWeight(weights[-1])
                out = []

                for i in range(self.NetworkShape[-1]):
                    c = last[i]
                    

                    for weight in reverse:
                        c = numpy.sum(Multiply(weight, c),axis=1)

                    out.append(c)
                result = [numpy.sum(result*o,axis=0) for o in out]
                print(result)

            result = numpy.sum(result*costs)
        
        print(SelfActivation)
        output = SelfActivation*result
        print(output)
        return output

    def forward(self,inputs:list):
        if (len(inputs) != self.NetworkShape[0]): raise ValueError("Wrong input counts")
        self.reset_activation()

        self.activations[0] = inputs
        for i in range(len(self.activations[0:-1])):
            act = self.activations[i]
            weights = self.weights[i]

            active = sigmoid((numpy.sum(RotateWeight(weights)*act,axis=1)))

            self.activations[i+1] = active



    def backpropgation(self, RValue:list):
        if (len(RValue) != self.NetworkShape[-1]): raise ValueError("Worng Counts of 'RValue' (=RValue)")

        cost = CostFunctionDerivative(self.activations[-1],RValue)

        for l, layer in enumerate(self.weights[0:-1]):
            for n, neuron in enumerate(layer):
                for w, weight in enumerate(neuron):
                    active = self.activations[l][n]

                    print(l,n,w)

                    weights = [self.weights[l+1][w]] + list(self.weights[l+2:])
                    print(f"cost: {cost}")
                    dw = self.PartialDerivative(weights, cost, active)

                    self.weights[l][n][w] = round(self.weights[l][n][w] - self.weights[l][n][w]*dw*self.l_rate, self.round)

import random

count = 1

net = network(0.3,[3,4,4,1])

net.forward([1,1,1])
print(numpy.sum(CostFunction(net.activations[-1],[0.3])))

for i in range(count):
    a = numpy.array([random.random() for i in range(3)])
    net.forward(a)
    net.backpropgation([numpy.sum(a*2)])

net.forward([1,1,1])
print(numpy.sum(CostFunction(net.activations[-1],[0.3])))
print(net.weights)
