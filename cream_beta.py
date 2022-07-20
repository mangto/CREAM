import numpy

def sigmoid(value):
    return 1/(1+numpy.exp(-1*value))

def RotateWeight(weights:numpy.array):
    return numpy.flip(numpy.rot90(weights,k=-1),1)

def MultiplyAll(a, b):
    return RotateWeight(numpy.tile(a,(len(b),1)))*b

def CostFunctionDerivative(output:list, real:list):
    out = numpy.array(output)
    real = numpy.array(real)

    cost = 2*(out-real)

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

    def PartialDerivative(self,weights:list):
        weights = list(reversed(weights))
        OutCount = self.NetworkShape[-1]
        result = [] #assume cost is 1 -> multiply later
        dummy = []

        for weight in weights[1:-1]:
            rotated = RotateWeight(weight)
            print(f'#rotated weight\n{rotated}')

        print("="*30)

        result = weights[-1]*1

        return result

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

        for l, layer in enumerate(self.weights[0:-1]):
            for n, neuron in enumerate(layer):
                for w, weight in enumerate(neuron):

                    active = self.activations[l][n]
                    weights = [self.weights[l+1][w]] + list(self.weights[l+2:])

                    dw = active*(cost*self.PartialDerivative(weights))
        
count = 1

net = network(0.6,[2,3,3,2,2,2,2])

for i in range(count):
    a = [1,0]
    net.forward(a)
    net.backpropgation([0,1])
