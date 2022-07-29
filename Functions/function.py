import numpy

def CostFunction(output, target, dervative:bool=False):
    if (dervative == False): return (numpy.array(output)-numpy.array(target))**2/2
    else: return numpy.array(output)-numpy.array(target)

def MultiplyEach(a:numpy.array,b:numpy.array): # MultiplyEach( [[1, 2], [3, 4]],   [1, 2] ) -> [[1, 2], [6, 8]]
    if (len(a) != len(b)): raise ValueError(f"Different inputs {len(a)}, {len(b)}")
    return numpy.array([a[i]*numpy.array(b[i]) for i in range(len(a))])
