import numpy

def sigmoid(value):
    return 1/(1+numpy.exp(-1*value))

def RotateWeight(weights:numpy.array):
    return numpy.flip(numpy.rot90(weights,k=-1),1)

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

            active = RotateWeight([a*rotated_weights[j] for j, a in enumerate(act)])
            active = [sigmoid(numpy.sum(a)) for a in active]

            self.activations[i+1] = active


    def backpropgation(self, RValue:list):
        if (len(RValue) != self.NetworkShape[-1]): raise ValueError("Worng Counts of 'RValue' (=RValue)")

        errors = [[(self.activations[-1][i]-RValue[i])**2/2 for i in range(self.NetworkShape[-1])]]
        new_weights = []
        ReversedActiv = list(reversed(self.activations))

        for l, neuron in enumerate(reversed(self.weights)):
            w = []
            error = []
            for i, weights in enumerate(neuron):
                w.append((weights*errors[l][i]*self.l_rate)+weights)
                error.append((numpy.array(ReversedActiv[l+1])-numpy.array(ReversedActiv[l+1])*errors[l][i])**2/2)
            error = RotateWeight(error)
            errors.append(numpy.sum(error,axis=1)/len(error[0]))

            new_weights.append(numpy.array(w))

        new_weights = list(reversed(new_weights))
        '''print("="*30)
        print(self.weights)
        print(new_weights)'''

        self.weights = new_weights



        
        return errors
count = 10000

net = network(0.6,[1,2,1])

import random

for i in range(count):
    a = random.randint(0,100)/100
    net.forward([a])
    net.backpropgation([a*3.14])

net.forward([0.3])
print(net.weights)
print(net.activations)
