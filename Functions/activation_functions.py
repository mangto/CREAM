import numpy

# Activation Functions
def sigmoid(value, Derivative=False):
    if (Derivative == False): return 1/(1+numpy.exp(-1*value))
    else:
        sig = sigmoid(value)
        return sig*(1-sig)

def ReLU(value, Derivative=False):
    if (Derivative == False): return max(0, value)
    else:
        if (value < 0): return 0
        else: return 1

def Leaky_ReLU(value, Derivative=False):
    if (Derivative == False): return max(0.01*value, value)
    else:
        if (value < 0): return 0.01
        else: return 1

def IdentityFunction(value, Derivative = False):
    if (Derivative == False): return value
    else: return 1