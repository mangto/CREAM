from cream.engine.deeplearning import network

def generate(sequence:list, lrate:float=0.01):
    '''
    generate network with sequence of layer
    network will be automatically compiled, so you don't need to compile
    '''

    net = network(lrate)
    for layer in sequence: net.add(layer)
    net.compile()
    return net