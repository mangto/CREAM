import numpy, random

import Csys # -> for system control
import visualizer
from Functions import * # -> afunctions
import datasets as dataset
import nvdia_smi


class snn:
    # 
    # SNN is an abbreviation of Simple Neural Network.
    # It has only one hidden layer
    # This is made to vertify if delta rule works
    # 
    # If you're reading this, you're on my github 
    # if not, please contact mangto0701@gmail.com
    # 

    def init_weights(NetworkShape:list):
        result = [
            [
                list(numpy.random.randn(NetworkShape[i+1])) for j in range(n)
            ] for i, n in enumerate(NetworkShape[:-1])
        ]

        return result

    def init_bias(LayerCount):
        result = list(numpy.random.randn(LayerCount-1))

        return result

    def reset_activation(self):
        return numpy.array([numpy.zeros(hn) for hn in self.NetworkShape],dtype=object)    

    def __init__(self, NetworkShape:list, function:sigmoid, LearningRate:float=0.3):

        if (len(NetworkShape) != 3): raise ValueError("That's not 'snn' a.k.a. Simple Neural Network")

        self.NetworkShape = NetworkShape
        self.function = function
        self.l_rate = LearningRate

        self.layercount = len(NetworkShape)
        self.weights = snn.init_weights(NetworkShape)
        self.biases = snn.init_bias(self.layercount)

        self.activations = self.reset_activation()
        self.derive_activations = self.reset_activation()

        self.actives = self.reset_activation()
        self.dactivations = self.reset_activation()

    def train(self, PackCount:int, dataset:list, MaxEpoch:int=None, EndCost:float=0.01):
        cost = 1
        epoch = 0

        while (cost > EndCost or epoch == 0 or (MaxEpoch != None and epoch < MaxEpoch)):
            epoch += 1
            target = numpy.zeros(self.NetworkShape[-1])
            actives = self.reset_activation()
            dactive = self.reset_activation()

            for i in range(PackCount):
                data = random.sample(dataset, 1)[0]
                self.forwardfeed(data[0])
                target = target + data[1]
                actives = actives + self.activations
                dactive = dactive + self.derive_activations

            actives /= PackCount
            dactive /= PackCount
            target = list(target/PackCount)

            self.actives = actives
            self.dactivations = dactive

            self.backpropgation(target)


            #test
            cost = 0
            for data in dataset:
                self.forwardfeed(data[0])
                cost += sum(Error(self.activations[-1],data[1]))
            Csys.out(f"{data} | {cost} | {self.biases}",Csys.bcolors.FAIL)


    def forwardfeed(self, input:list):
        if (type(input) != list and type(input) != numpy.array): raise TypeError(f"We need list or numpy.array for input, not {type(input)}")
        if (len(input) != self.NetworkShape[0]): raise ValueError("nah,, wrong count of input")

        self.activations = self.reset_activation()
        self.pure_activations = self.reset_activation()
        
        self.activations[0] = input
        self.derive_activations[0] = input

        for i in range(self.layercount-1):
            new_activ = MultiplyEach(self.weights[i],self.activations[i])
            new_activ = numpy.sum(new_activ,axis=0)+self.biases[i]
            self.derive_activations[i+1] = [self.function(i, Derivative=True) for i in new_activ]
            self.activations[i+1] =[self.function(i) for i in new_activ]

    def backpropgation(self, target:list): # target is a list of numbers that our network have to make out
        if (type(target) != list and type(target) != numpy.array): raise TypeError(f"We need list or numpy.array for input, not {type(input)}")
        if (len(target) != self.NetworkShape[-1]): raise ValueError("nah,, wrong count of input")

        Errors = Error(self.actives[-1], target)
        ErrorTotal = sum(Errors)
        DErrors = Error(self.actives[-1], target, True)
        

        dws = DErrors*numpy.array(self.derive_activations[-1])
        # Partial Derivative in 1st layer of weight
    
        for n, neuron in enumerate(self.weights[0]):
            for w, weight in enumerate(neuron):
                self.weights[0][n][w] = self.weights[0][n][w]*(1-sum(dws*self.l_rate*self.actives[0][n]*self.weights[1][w]*self.dactivations[1][w]))

        # Partial Derivative in 2nd layer of weight


        for n, neuron in enumerate(self.actives[-2]):
            dw = dws*neuron
            self.weights[1] = self.weights[1]*(1-dw*self.l_rate)


class network:
    def init_weights(NetworkShape:list):
        result = [
            [
                list(numpy.random.randn(NetworkShape[i+1])) for j in range(n)
            ] for i, n in enumerate(NetworkShape[:-1])
        ]

        return result

    def init_bias(LayerCount):
        result = list(numpy.random.randn(LayerCount-1))

        return result

    def __init__(self, NetworkShape:list,function=sigmoid,LearningRate:float=0.3):
        self.NetworkShape = NetworkShape
        self.function = function
        self.l_rate = LearningRate

        self.layercount = len(NetworkShape)
        self.weights = network.init_weights(NetworkShape)
        self.biases = network.init_bias(self.layercount)

        self.activations = self.reset_activation()
        self.pure_activations = self.reset_activation()

    def reset_activation(self):
        return numpy.array([numpy.zeros(hn) for hn in self.NetworkShape],dtype=object)     

    def forward(self, input:list):
        if (type(input) != list and type(input) != numpy.array): raise TypeError(f"We need list or numpy.array for input, not {type(input)}")
        if (len(input) != self.NetworkShape[0]): raise ValueError("nah,, wrong count of input")

        self.activations = self.reset_activation()
        self.pure_activations = self.reset_activation()
        
        self.activations[0] = input
        self.pure_activations[0] = input

        for i in range(self.layercount-1):
            new_activ = MultiplyEach(self.weights[i],self.activations[i])
            new_activ = numpy.sum(new_activ,axis=0)+self.biases[i]
            self.pure_activations[i+1] = new_activ
            new_activ = [self.function(i) for i in new_activ]
            self.activations[i+1] = new_activ

    def backward(self):

        pass