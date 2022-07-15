import numpy

def sigmoid(value):
    return 1/(1+numpy.exp(-1*value))

def RotateWeight(weights:numpy.array):
    return numpy.flip(numpy.rot90(weights,k=-1),1)

class network:
    def __init__(self, NetworkShape:list):
        self.NetworkShape = NetworkShape
        self.weights = [numpy.array([list(numpy.random.randn(i)) for j in range(NetworkShape[k+1])]) for k, i in enumerate(NetworkShape[0:-1])]
        self.activations = self.reset_activation()

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


    def backward(self, RValue:list):
        if (len(RValue) != self.NetworkShape[-1]): raise ValueError("Worng Counts of RValue (=RValue)")

        errors = [self.activations[-1][i] - RValue[i] for i in range(self.NetworkShape[-1])]

        print(errors)

net = network([10,5,3,2])
net.forward([1,2,3,4,5,6,7,8,9,10])
net.backward([1,0])
