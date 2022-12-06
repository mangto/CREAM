from cream.engine.deeplearning import network
from cream.layer.dense import Dense
from cream.functions import *

def generate(shapes:list, activation=None, lrate:float=0.01):
    '''
    generate network with shape of dense layer
    '''
    net = network(lrate)
    activation = activation if activation else ReLU

    for i, shape in enumerate(shapes[1:]):
        if (i == 0):
            net.add(Dense(shape, activation, shapes[0]))
            continue
        net.add(Dense(shape, activation))
    net.compile()
    return net