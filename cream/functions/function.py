import numpy

def Error(output, target, dervative:bool=False):
    if (dervative == False): return (numpy.array(output)-numpy.array(target))**2/2
    else: return numpy.array(output)-numpy.array(target)