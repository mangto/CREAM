import cream.tool.Csys as Csys # -> for system control
from cream.Functions import * # -> afunctions

import random

class onsnn:
    input_type = [int, float]

    def check(self):
        Csys.out(f"weights: {self.weights}", Csys.bcolors.OKBLUE)
        Csys.out(f"biases: {self.biases}", Csys.bcolors.OKBLUE)


    def __init__(self, activation_function=sigmoid, learning_rate=0.01):
        self.weights = numpy.random.randn(3)*0.1
        self.biases = numpy.zeros(3)
        self.acfunc = activation_function
        self.lrate = learning_rate

        self.activations = [0. ,0. ,0. ,0.]
        self.ractivations = [0. ,0. ,0. ,0.]

    def forward(self, input):
        assert type(input) in onsnn.input_type , "Wrong Type of Input"

        output = input
        self.activations = [input]
        self.ractivations = [input]

        for weight, bias in zip(self.weights, self.biases):
            
            routput = output*weight + bias
            output = self.acfunc(routput)

            self.activations.append(output)
            self.ractivations.append(routput)

        return output

    def backward(self, target):
        assert type(target) in onsnn.input_type , "Wrong Type of Target"

        delta = self.activations[3] - target

        # h2 -> out
        delta_h2o = delta * self.acfunc(self.ractivations[2], True)

        self.weights[2] -= self.lrate * delta_h2o * self.activations[2]
        self.biases[2] -= self.lrate * delta_h2o

        # h1 -> h2
        delta_h1h2 = delta_h2o * self.weights[2] * self.acfunc(self.ractivations[1], True)

        self.weights[1] -= self.lrate * delta_h1h2 * self.activations[1]
        self.biases[1] -= self.lrate * delta_h1h2

        # in -> h1
        delta_ih1 = delta_h1h2 * self.weights[1] * self.acfunc(self.ractivations[0], True)

        self.weights[0] -= self.lrate * delta_ih1 * self.activations[0]
        self.biases[0] -= self.lrate * delta_ih1
