import numpy

# Activation Functions
def sigmoid(value, Derivative=False):
    if (Derivative == False): return 1/(1+numpy.exp(-1*value))
    else:
        sig = sigmoid(value)
        return sig*(1-sig)

def ReLU(value, Derivative=False):
    if (Derivative == False): return numpy.maximum(0, value)
    else:
        value[value<=0] = 0
        value[value>0] = 1
        return value

def Leaky_ReLU(value, Derivative=False):
    if (Derivative == False): return numpy.maximum(0.01*value, value)
    else:
        value[value<=0] = 0.01
        value[value>0] = 1
        return value

def Linear(value, Derivative = False):
    if (Derivative == False): return value
    else: return 1