import numpy
import cream.tool.csys as csys
import cream.functions as functions

class network:

    InputType = [list, numpy.ndarray]
    def __init__(self, lrate:float=0.01):
        '''
        initialize neural network

        .add(layer:Layer): add layer to network
        .compile(): compile neural network
            * Determine weights and biases
            * If you don't, network won't work
        '''
        self.lrate = lrate

        self.layers:list= []
        self.shape:list= []
        self.depth:int= 0
        self.compiled : bool = False
        self.weights:list = []
        self.biases:list = []

        self.activ = network.reset_activation(self.shape)
        self.raw_activ = network.reset_activation(self.shape)


    def reset_activation(NetworkShape):
        # reset activations

        result = [numpy.zeros(shape) for shape in NetworkShape]

        return result

    def add(self, layer):
        '''
        add layer to network
        '''

        self.depth += 1
        self.layers.append(layer)

        return

    def compile(self) -> None:
        '''
        compile neural network

        network must be compiled because weights and biases are determined through this funciton.
        '''
        try:

            for i, layer in enumerate(self.layers):
                if (i == 0 and not layer.InputShape): raise ValueError("No input layer") 
                if (layer.InputShape): self.shape.append(layer.InputShape)
                layer.generate(self.shape[i])
                self.shape.append(layer.size)

                # 레이어별로 generate함수 만들기
                # conv도 생각할것!
                # flatten은 init도 안만듦

            self.depth += 1
        except Exception as e:
            csys.error(e, "Compile Error")

        self.compiled : bool = True

        return

    def forward(self, input:list|numpy.ndarray) -> list | numpy.ndarray:
        '''
        feed network forward

        It saves raw-activation (not activation-functioned) and activation for backpropagation
        '''

        assert self.compiled, "Network not compiled"
        assert type(input) in network.InputType, "Wrong Type of Input"
        assert len(input) == self.shape[0], f"Wrong Count of Input, need: {self.shape[0]} taken: {len(input)}"

        self.activ = network.reset_activation(self.shape)
        self.raw_activ = network.reset_activation(self.shape)
        self.activ[0] = input
        self.raw_activ[0] = input

        for i, layer in enumerate(self.layers):
            raw, refined = layer.do(self.activ[i])
            self.raw_activ[i+1] = raw
            self.activ[i+1] = refined

        return self.activ[-1]

    def __call__(self, input:list|numpy.ndarray) -> list | numpy.ndarray:
        return self.forward(input)

    def backward(self, target:list|numpy.ndarray, activations=None, raw_activations=None) -> float:
        '''
        backpropagation of network
        '''

        assert self.compiled, "Network not compiled"
        assert type(target) in network.InputType, "Wrong Type of Target"
        assert len(target) == self.shape[-1], f"Wrong Count of Target, need: {self.shape[-1]} taken: {len(target)}"

        activations = activations if activations else self.activ
        raw_activations = raw_activations if raw_activations else self.raw_activ

        error = functions.Error(activations[-1], target)
        derror = functions.Error(activations[-1], target, True)
        delta = derror

        for i in range(self.depth - 1):
            index :int = - i - 1
            args :dict = {'delta':delta, 'layer':self.layers, 'index':index, 'lrate':self.lrate, 
                            'activation':self.activ, 'raw_activation':self.raw_activ}
            returned :dict = self.layers[index].backpropagation(args)
            delta = returned.get('delta', delta)

        return error 

    def fit(self, inputs: list | numpy.ndarray, targets: list | numpy.ndarray,
            MinError:int=None, MaxEpoch:int=None,
            ForwardFunction=None, LearningFunction=None,
            advise=True):

        '''
        fit network by repeating forwarding and backwarding
        '''

        assert (icount := len(inputs)) == (tcount := len(targets)), "Different count of inputs and targets"

        ForwardFunction = ForwardFunction if ForwardFunction else self.forward
        LearningFunction = LearningFunction if LearningFunction else self.backward

        error = None
        epoch = 0

        def is_valid(error, epoch, MinError=None, MaxEpoch=None):
            '''
            check if network have to continue learning
            '''

            MinError = MinError if MinError or MinError == 0 else None
            MaxEpoch = MaxEpoch if MaxEpoch or MaxEpoch == 0 else None
            if (MinError == None and MaxEpoch == None): csys.error('Both MinError and MaxEpoch are None', 'Infinite loop')
            if (error == None): return True

            if (MinError != None and MaxEpoch == None):
                if (error > MinError): return True
                return False
            elif (MinError == None and MaxEpoch != None):
                if (epoch < MaxEpoch): return True
                return False
            else:
                if (epoch >= MaxEpoch): return False
                if (error <= MinError): return False
                return True

        valid = is_valid(error, epoch, MinError, MaxEpoch)

        while valid:
            error = 0

            for input, target in zip(inputs, targets):
                ForwardFunction(input)
                LearningFunction(target)

                error += numpy.sum(functions.Error(self.activ[-1], target))
            

            epoch += 1
            valid = is_valid(error, epoch, MinError, MaxEpoch)
            if (advise): print(f"{epoch=:>6} | {error=}")

        return