import numpy

import cream.tool.Csys as Csys # -> for system control
from cream.Functions import * # -> afunctions

class network:

    InputType = [list, numpy.array, numpy.ndarray]

    def __str__(self):
        result = f'''
        | type: Cream Neural Network
        | Network Type: Normal
        | Network Shape: {self.NetworkShape}
        | Activation Function: {self.acfunc}
        | Learning Rate: {self.lrate}
        '''

        return result

    def check(self):
        result = f'''
        weights: {self.weights}
        biases : {self.biases}
        '''
        return result

    def init_weights(NetworkShape:list):
        # initialize weights with network shape

        result = [[numpy.random.randn(NetworkShape[i])* 0.1 for j in range(shape)]
                    for i, shape in enumerate(NetworkShape[1:])] 

        return result

    def init_biases(NetworkShape:list):
        # initialize biases with network shape

        result = [numpy.random.randn(shape)*0.1 for shape in NetworkShape[1:]]

        return result

    def reset_activation(NetworkShape):
        # reset activations

        result = [numpy.zeros(shape) for shape in NetworkShape]

        return result
    

    def __init__(self, NetworkShape:list, ActivationFunction=sigmoid, LearningRate:float=0.3,
                    weights:numpy.array=None, biases:numpy.array=None):
        self.NetworkShape = NetworkShape
        self.acfunc = ActivationFunction
        self.lrate = LearningRate

        self.weights = weights if weights else network.init_weights(NetworkShape)
        self.biases = biases if biases else network.init_biases(NetworkShape)

        self.activ = network.reset_activation(NetworkShape)
        self.raw_activ = network.reset_activation(NetworkShape)

        self.depth = len(NetworkShape)
    def forward(self, input:list):
        assert type(input) in network.InputType, "Wrong Type of Input"
        assert len(input) == self.NetworkShape[0], f"Wrong Count of Input, need: {self.NetworkShape[0]} taken: {len(input)}"

        self.activ = network.reset_activation(self.NetworkShape)
        self.raw_activ = network.reset_activation(self.NetworkShape)
        self.activ[0] = input
        self.raw_activ[0] = input

        for i in range(len(self.NetworkShape[1:])):
            raw = numpy.sum(numpy.array(self.weights[i]) * numpy.array(self.activ[i]), axis=1) + self.biases[i]

            self.raw_activ[i+1] = raw
            self.activ[i+1] = self.acfunc(raw)
    
    def backpropgation(self, target:list, activations=None, raw_activations=None):
        assert type(target) in network.InputType, "Wrong Type of Target"
        assert len(target) == self.NetworkShape[-1], f"Wrong Count of Input, need: {self.NetworkShape[-1]} taken: {len(target)}"

        activations = activations if activations else self.activ
        raw_activations = raw_activations if raw_activations else self.raw_activ

        # Code Of SNN
        # 
        # delta = numpy.array(self.activations[-1]) - target
        # self.weights[1] -= self.l_rate * numpy.transpose(numpy.reshape(delta, (self.NetworkShape[-1], 1)) * self.activations[1])
        # self.biases[1] -= self.l_rate * delta

        # h = numpy.array(self.activations[1])
        # delta = numpy.sum(self.weights[1] * delta,axis=1)*h*(1-h)
        # self.weights[0] -= self.l_rate * numpy.transpose(numpy.reshape(delta, (self.NetworkShape[1], 1)) * self.activations[0])
        # self.biases[0] -= self.l_rate * delta

        error = activations[-1] - target

        delta = error
        for l, layer in enumerate(reversed(self.weights)):
            
            l = self.depth-2-l # reverse sequence


            if (l < self.depth-2): # if layer is not hidden_last to output
                weight = numpy.transpose(self.weights[l+1])
                delta = numpy.sum(delta * weight, axis=1)/len(weight)* self.acfunc(raw_activations[l+1],True)
                # Csys.out(l, Csys.bcolors.FAIL)
            
            else:
                delta = delta * self.acfunc(raw_activations[l+1], True)
 
            dw = numpy.dot(delta[:,None], numpy.array(self.activ[l])[None])

            # print(dw)
            # Csys.division(30)

            self.weights[l] -= dw * self.activ[l] * self.lrate  * self.weights[l]
            self.biases[l] -= numpy.sum(dw, axis=1)* self.lrate * self.biases[l]

        #     Csys.out(self.activ[l], Csys.bcolors.OKBLUE)
        #     print(dw * self.activ[l])
        #     Csys.division(60)

        # Csys.stop()


    def backward(self, target:list, activations=None, raw_activations=None):
        assert type(target) in network.InputType, "Wrong Type of Target"
        assert len(target) == self.NetworkShape[-1], f"Wrong Count of Input, need: {self.NetworkShape[-1]} taken: {len(target)}"

        activations = activations if activations else self.activ
        raw_activations = raw_activations if raw_activations else self.raw_activ

        error = Error(activations[-1] - target)
        delta = error

        for l in range(self.depth-1):
            l = self.depth - 2 - l # Reverse Sequence 


    def train(self, datasets, MaxEpoch:int=None, MinError:float=None):
        error = 0
        epoch = 0
        LastError = 0 # to disadventage error of ReLU


        errorTF = MinError and error > MinError
        while ((MaxEpoch and epoch < MaxEpoch or (errorTF or not MinError)) or epoch == 0):

            error = 0
            epoch += 1
            for data in datasets:
                if (type(data[0]) in [int, float]): data[0] = [data[0]]
                if (type(data[1]) in [int, float]): data[1] = [data[1]]

                self.forward(data[0])
                self.backpropgation(data[1])

                error += sum(Error(self.activ[-1],data[1],))

            Csys.out(f"{epoch} | {error}", Csys.bcolors.OKCYAN)
            if (abs(LastError) - abs(error) <= 0 ):
                Csys.out(f"| network changed", Csys.bcolors.FAIL)
                self.weights = network.init_weights(self.NetworkShape)
                self.biases = network.init_biases(self.NetworkShape)


            errorTF = MinError and abs(error) > MinError
            LastError = error
                
