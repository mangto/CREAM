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
        self.biases = numpy.zeros(self.layercount-1) #snn.init_bias(self.layercount)

        self.activations = self.reset_activation()
        self.derive_activations = self.reset_activation()

        self.actives = self.reset_activation()
        self.dactivations = self.reset_activation()

    def train(self, PackCount:int, dataset:list, MaxEpoch:int=None, EndCost:float=None):
        cost = 1
        epoch = 0

        while ((EndCost != None and cost > EndCost) or epoch == 0 or (MaxEpoch != None and epoch < MaxEpoch)):
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
            Csys.out(f"{epoch} {cost}",Csys.bcolors.FAIL)
