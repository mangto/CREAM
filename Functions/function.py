import numpy

def Error(output, target, dervative:bool=False):
    if (dervative == False): return (numpy.array(output)-numpy.array(target))**2/2
    else: return numpy.array(output)-numpy.array(target)

def MultiplyEach(a:numpy.array,b:numpy.array): # MultiplyEach( [[1, 2], [3, 4]],   [1, 2] ) -> [[1, 2], [6, 8]]
    if (len(a) != len(b)): raise ValueError(f"Different inputs {len(a)}, {len(b)}")
    return numpy.array([a[i]*numpy.array(b[i]) for i in range(len(a))])

def MultiplyEach3(a:numpy.array,b:numpy.array,c:numpy.array): # MultiplyEach( [[1, 2], [3, 4]],   [1, 2] ) -> [[1, 2], [6, 8]]
    if (len(a) != len(b)): raise ValueError(f"Different inputs {len(a)}, {len(b)}, {len(c)}")
    return numpy.array([a[i]*numpy.array(b[i])*numpy.array(c[i]) for i in range(len(a))])

def Pad(value:numpy.array, padd=2):
    numpy.pad(value,((padd,padd),(padd,padd)), 'constant', constant_values=0)
    