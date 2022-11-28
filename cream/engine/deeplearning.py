import numpy
import cream.tool.csys as csys

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

    def foreward(self, input:list|numpy.ndarray) -> list | numpy.ndarray:
        '''
        feed network forward

        It saves raw-activation (not activation-functioned) and activation for backpropagation
        '''
        assert type(input) in network.InputType, "Wrong Type of Input"
        assert len(input) == self.shape[0], f"Wrong Count of Input, need: {self.shape[0]} taken: {len(input)}"

        try:
            self.activ = network.reset_activation(self.shape)
            self.raw_activ = network.reset_activation(self.shape)
            self.activ[0] = input
            self.raw_activ[0] = input

            for i, layer in enumerate(self.layers):
                raw, refined = layer.do(self.activ[i])
                self.raw_activ[i+1] = raw
                self.activ[i+1] = refined

            return self.activ[-1]

        except Exception as e:
            csys.out(f"Errored layer: {i}", csys.WARNING)
            csys.error(e, "While Forwarding")

    def backward(self, target:list|numpy.ndarray, activations=None, raw_activations=None) -> None:

        activations = activations if activations else self.activ
        raw_activations = raw_activations if raw_activations else self.raw_activ

        error = activations[-1] - target
        delta = error

        return