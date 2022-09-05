import numpy

# Activation Functions
def sigmoid(value, Derivative=False):
    value = numpy.array(value)

    if (Derivative == False): return 1/(1+numpy.exp(-1*value))
    else:
        sig = sigmoid(value)
        return sig*(1-sig)

def msigmoid(value, Derivative=False, Multiplier:float=2.0):
    value = numpy.array(value)

    if (not Derivative):
        return sigmoid(value)*Multiplier-Multiplier/2

    else:
        sig = sigmoid(value)
        return sig*(1-sig) * Multiplier

def ReLU(value, Derivative=False):
    value = numpy.array(value)

    if (Derivative == False): return numpy.maximum(0, value)
    else:
        value[value<=0] = 0
        value[value>0] = 1
        return value

def Leaky_ReLU(value, Derivative=False):
    value = numpy.array(value)

    if (Derivative == False): return numpy.maximum(0.01*value, value)
    else:
        value = numpy.array(value)
        value[value<=0] = 0.01
        value[value>0] = 1
        return value

def Linear(value, Derivative = False):
    value = numpy.array(value)

    if (Derivative == False): return value
    else: return numpy.ones(value.shape)