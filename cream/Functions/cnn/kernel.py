import numpy

def gaussian(shape=(3,3)):
    return numpy.random.normal(size=shape)

class box:
    def __init__(self, shape=(3,3)):
        self.kernel = numpy.ones(shape)
        self.multiplier = 1/numpy.sum(self.kernel)
        self.shape = shape
    
    def __str__(self):
        return str(self.kernel)
        

roberts_1 = numpy.array([[1, 0],
                         [0,-1]])

roberts_2 = numpy.array([[0, 1],
                         [-1,0]])


sobel_x = numpy.array([ [-1,  0,  1],
                        [-2,  0,  2],
                        [-1,  0,  1] ])

sobel_y = numpy.array([ [ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1] ])


prewitt_x = numpy.array([ [-1,  0,  1],
                          [-1,  0,  1],
                          [-1,  0,  1] ])

prewitt_y = numpy.array([ [ 1,  1,  1],
                          [ 0,  0,  0],
                          [-1, -1, -1] ])


LoG_3_1 = numpy.array([ [ 0, -1,  0],
                        [-1,  4, -1],
                        [ 0, -1,  0] ])

LoG_3_2 = numpy.array([ [-1, -1, -1],
                        [-1,  8, -1],
                        [-1, -1, -1] ])

LoG_5 = numpy.array([ [ 0,  0, -1,  0,  0],
                      [ 0, -1, -2, -1,  0],
                      [-1, -2, 16, -2, -1],
                      [ 0, -1, -2, -1,  0],
                      [ 0,  0, -1,  0,  0] ])

LoG_9 = numpy.array([ [ 0,  1,  1,  2,  2,  2,  1,  1,  0],
                      [ 1,  2,  4,  5,  5,  5,  4,  2,  1],
                      [ 1,  4,  5,  3,  0,  3,  5,  4,  1],
                      [ 2,  5,  3,-12,-24,-12,  3,  5,  2],
                      [ 2,  5,  0,-24,-40,-24,  0,  5,  2],
                      [ 2,  5,  3,-12,-24,-12,  3,  5,  2],
                      [ 1,  4,  5,  3,  0,  3,  5,  4,  1],
                      [ 1,  2,  4,  5,  5,  5,  4,  2,  1],
                      [ 0,  1,  1,  2,  2,  2,  1,  1,  0] ])

class threshold:
    roberts = 50
    sobel = 140
    prewitt = 100
    LoG_3_1 = 70
    LoG_3_2 = 150
    LoG_5 = 150
    LoG_9 = 2000