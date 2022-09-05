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

        result = [[numpy.random.normal(size=NetworkShape[i])* 0.1 for j in range(shape)]
                    for i, shape in enumerate(NetworkShape[1:])] 

        return result

    def init_biases(NetworkShape:list):
        # initialize biases with network shape

        # result = [numpy.random.randn(shape)*0.1 for shape in NetworkShape[1:]]
        result = [numpy.zeros(shape) for shape in NetworkShape[1:]]

        return result

    def reset_activation(NetworkShape):
        # reset activations

        result = [numpy.zeros(shape) for shape in NetworkShape]

        return result
        
    def load_weight(self, weight):
        self.weights = [numpy.array(w) for w in weight]

    def load_bias(self, bias):
        self.biases = [numpy.array(b) for b in bias]

    def train_advice(self, text, epoch, error, lrate):
        result = text.replace("/epoch/", str(epoch)).replace("/error/", str(error)).replace("/lrate/", str(lrate))
        return result

    def __init__(self, NetworkShape:list, ActivationFunction=sigmoid, LearningRate:float=0.3,
                    weights:numpy.array=None, biases:numpy.array=None):

        # assert len(NetworkShape) <= 4, f"Depth of Neural Network has to be 4, Current Depth: {len(NetworkShape)}"


        self.NetworkShape = NetworkShape
        self.acfunc = ActivationFunction
        self.lrate = LearningRate

        self.weights = weights if weights else network.init_weights(NetworkShape)
        self.biases = biases if biases else network.init_biases(NetworkShape)

        self.activ = network.reset_activation(NetworkShape)
        self.raw_activ = network.reset_activation(NetworkShape)

        self.depth = len(NetworkShape)
        self.start_nb = self.check()

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

        error = (activations[-1] - target)
        delta = error

        
        # Csys.out(self.activ, Csys.bcolors.FAIL)
        delta = delta * self.acfunc(self.raw_activ[-1], True)
        self.weights[-1] -= self.lrate * numpy.outer(delta, self.activ[-2])
        # print(self.lrate * numpy.outer(delta, self.activ[-2]))
        self.biases[-1] -= self.lrate * delta

        for i in range(self.depth - 2):
            delta = delta  * numpy.transpose(self.weights[-i-1])* self.acfunc(self.raw_activ[-i-2], True)[:,None]
            delta = numpy.sum(delta, axis=1) / len(delta[0])
            self.weights[-i-2] -= self.lrate * numpy.outer(delta, self.activ[-i-3])
            self.biases[-i-2] -= self.lrate * delta
        

    def train(self, datasets, MaxEpoch:int=None, MinError:float=None,
                lrate_change=0, Min_lrate:float=None, Max_lrate:float=None,
                advice="/epoch/ | /error/"):
        error = 0
        epoch = 0
        LastError = 0 # to disadventage error of ReLU


        errorTF = MinError and error > MinError
        while ((MaxEpoch and epoch < MaxEpoch or (errorTF or not MinError)) or epoch == 0):
            self.lrate += lrate_change
            if (Min_lrate != None): self.lrate = max(self.lrate, Min_lrate)
            if (Max_lrate != None): self.lrate = min(self.lrate, Max_lrate)

            error = 0
            epoch += 1
            for data in datasets:
                if (type(data[0]) in [int, float]): data[0] = [data[0]]
                if (type(data[1]) in [int, float]): data[1] = [data[1]]

                self.forward(data[0])
                self.backpropgation(data[1])

                error += sum(Error(self.activ[-1],data[1],))

            
            Csys.out(self.train_advice(advice,epoch,error,self.lrate), Csys.bcolors.OKCYAN)
            # if(LastError - error == 0):
            #     Csys.out(self.check(), Csys.bcolors.OKBLUE)
            #     Csys.out(self.activ,Csys.bcolors.OKGREEN)
            #     Csys.stop()
            # if (abs(LastError) - abs(error) <= 0 ):
            #     Csys.out(f"| network changed", Csys.bcolors.FAIL)
            #     self.weights = network.init_weights(self.NetworkShape)
            #     self.biases = network.init_biases(self.NetworkShape)


            errorTF = MinError and abs(error) > MinError
            LastError = error

        Csys.out(self.check(), Csys.bcolors.OKBLUE)