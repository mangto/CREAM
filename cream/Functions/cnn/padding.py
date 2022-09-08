import numpy

def pad(value:numpy.array, padd=2):
    return numpy.pad(value,((padd,padd),(padd,padd)), 'constant', constant_values=0)
