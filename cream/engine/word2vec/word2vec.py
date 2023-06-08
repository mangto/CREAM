import numpy, numba
import cream.tool.csys as csys
import cream.functions as functions
from cream.layer.dense import Dense


import time


def generate_one_hot(id:int, size:int):
    vector = numpy.zeros((size, ))
    vector[id] = 1

    return vector

class network:

    InputType = [list, numpy.ndarray]
    def __init__(self, VectorSize:int, WindowSize:int=2, dimension:int=300, lrate:float=0.01):
        '''
        initialize neural network
        '''
        self.WindowSize = WindowSize
        self.dimension = dimension
        self.lrate = lrate

        self.train_data_cache = {}

        self.weights = [
            numpy.random.uniform(-1.0, 1.0, (dimension, VectorSize)) * 0.1,
            numpy.random.uniform(-1.0, 1.0, (VectorSize, dimension)) * 0.1
        ]

        self.activation:list

    def generate_train_data(self, tokens:list, word2id:dict) -> tuple[numpy.ndarray, numpy.ndarray]:
        targets = []
        surroundings = []
        one_hot_size = len(word2id)

        for i, data in enumerate(tokens):
            token = data["ids"]

            for i, target in enumerate(token):
                surrounding = token[max(0, i-self.WindowSize):i] + token[i+1:i+1+self.WindowSize]
                count = len(surrounding)

                surroundings += surrounding
                targets += [target] * count

        count = len(targets)
        target = numpy.zeros((count, one_hot_size))
        target[numpy.arange(count), numpy.array(targets)] = 1
        surrounding = numpy.zeros((count, one_hot_size))
        surrounding[numpy.arange(count), numpy.array(surroundings)] = 1

        del surroundings
        del targets

        return target, surrounding
    
    def extract(self, id, word2id):
        sdr = numpy.stack((generate_one_hot(id, len(word2id)),))
        self.forward(sdr)

        return self.activation[0][0]

    def forward(self, surrounding):
        h = surrounding @ self.weights[0].T
        out = h @ self.weights[1].T
        rout = functions.softmax2d(out)

        self.activation = [h, out, rout]

    def backward(self, surrounding, target):


        delta = self.activation[-1] - target
        dw2 = (self.activation[0].T @ delta).T
        delta = delta @ self.weights[1]

        dw1 = (surrounding.T @ delta).T

        self.weights[0] -= self.lrate * dw1 / len(dw1)
        self.weights[1] -= self.lrate * dw2 / len(dw2)
    
    def fit(self, corpus:list, word2id:dict, epoch:int, BatchSentence:int=10):

        '''
        fit network by repeating forwarding and backwarding
        '''

        STcount = len(corpus) # SenTence count

        for ep in range(epoch):
            print(f'----- [{ep}] -----')

            for i in range(STcount//BatchSentence + 1):
                start = time.time()
                if (BatchSentence * i == STcount): continue
                target, surrounding = self.generate_train_data(corpus[i*BatchSentence:(i+1)*BatchSentence], word2id)
                self.forward(surrounding)
                self.backward(surrounding, target)

                del target
                del surrounding

                print(f"[{i}] done! | estimated: {round(time.time()-start, 3)}s")


        
        return