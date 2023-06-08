
# cream trainer

from cream.engine.deeplearning import network
import cream.tool.csys as csys
import numpy

class trainer:
    def __init__(self, network:network, **settings) -> None:
        '''
        train network
        Maybe better than default 'fit' funciton of network

        [settings description]
        forward: Forward function of network.
        backward: Backward function (Backpropagation) of network.
        batch: Set batch count while training.
        '''

        self.network = network

        self.ForwardFunc = settings.get('forward', network.forward)
        self.BackwardFunc = settings.get('backward', network.backward)
        self.BatchCount = settings.get('batch', 1)

        pass

    def train(self,
              inputs: list|numpy.ndarray, targets: list|numpy.ndarray,
              MinError :int|float = None, MaxEpoch :int = None,
              **settings
              ) -> str:
        '''
        Return result of training.
        '''

        # warning section

        IgnoreWaring = settings.get('warn', False)

        assert (icount := len(inputs)) == (tcount := len(targets)), "Different count of inputs and targets."
        if (not IgnoreWaring):
            if (MaxEpoch == None): csys.warn("Infinite loop (MaxEpoch is not set)")
            if (MinError == None or MinError <= 0): csys.warn("Infinite loop (MinError have to be positive number)")

        # training section

        epoch :int = 0 # count of training
        error :int = None

        def is_valid(error, epoch, MinError=None, MaxEpoch=None) -> bool:
            if (error == None): return True
            if (MinError != None and error > MinError): return False
            if (MaxEpoch != None and epoch >= MaxEpoch): return False
            return True

        validity = is_valid(error, epoch, MinError, MaxEpoch)

        while validity:
            
            validity = is_valid(error, epoch, MinError, MaxEpoch)

        advice = 'Maybe advice?'

        return advice
    