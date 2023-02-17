import re, numpy, numba, pickle
from cream.functions.activation import softmax
from cream.tool import progress_bar


class network:
    def __init__(self, word2id, n_embedding, lrate:float=0.01):
        '''
        * n_embedding: result dimensions
        '''

        self.word2id:dict = word2id
        self.vocab_size = len(word2id)
        self.weights = [numpy.random.normal(size = (self.vocab_size, n_embedding), scale=0.001), 
                        numpy.random.normal(size = (n_embedding, self.vocab_size), scale=0.001)]
        self.lrate = lrate
    @numba.jit(forceobj=True)
    def forward(self, x, return_cache:bool = True):
        cache = {}

        cache['a1'] = x @ self.weights[0]
        cache['a2'] = cache['a1'] @ self.weights[1]
        cache['z'] = numpy.array([softmax(c) for c in cache['a2']])

        if not return_cache: return cache['z']
        return cache
    
    @numba.jit(forceobj=True)
    def backward(self, cache, x, y):
        delta = cache['z'] - y
        dw2 = cache['a1'].T @ delta
        delta = delta @ self.weights[1].T
        dw1 = x.T @ delta

        assert dw2.shape == self.weights[1].shape
        assert dw1.shape == self.weights[0].shape

        self.weights[0] -= self.lrate * dw1
        self.weights[1] -= self.lrate * dw2

        return cross_entropy(cache['z'], y)
    
    @numba.jit(forceobj=True)
    def train(self, x, y, BatchScale:int, MaxEpoch:int,
              ProgresBar:bool=True, PBlength:int=40,
              AutoSave=True, ModelFile='.\\model.cmdl') -> None:

        batch = int(len(x)/BatchScale)

        for i in range(MaxEpoch):
            if (ProgresBar):
                pb = progress_bar.bar(PBlength, design=progress_bar.box, title=f'train {i}')
                pb.start()
            n = 0
            for j in range(batch):
                n += 1
                if (ProgresBar): pb.update(n/batch*100)
                cache = self.forward(x[j*BatchScale:(j+1)*BatchScale])
                error = self.backward(cache, x[j*BatchScale:(j+1)*BatchScale], y[j*BatchScale:(j+1)*BatchScale])

            if (AutoSave): pickle.dump({"model":self,"x":x, "y":y, "word2id":self.word2id}, open(ModelFile, 'wb'))
            pb.end()

        return

    def GetEmbedding(self, word):
        id = self.word2id.get(word)

        if (id == None): raise KeyError(f'{word} not exist')
        
        OneHotVector = one_hot_encode(id, self.vocab_size)
        return self.forward(OneHotVector)['a1']




def cross_entropy(z, y):
    return - numpy.sum(numpy.log(z) * y)

def tokenize(text:str, lower:bool=True, RemoveSpecials:bool=True) -> list:
    '''
    tokenize text and return list

    * lower: lower text [ex) A -> a]
    * RemoveSpecials: remove special characters [ex) a!$b -> ab]
    '''

    assert type(text) == str, 'type of text must be str'

    if (lower): text = text.lower()
    if (RemoveSpecials): text = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", "", text)

    result = text.split(' ')

    return result

def map(tokens:list) -> tuple[dict, dict]:
    word2id = {}
    id2word = {}

    for i, token in enumerate(set(tokens)): # set은 겹치는게 존재할 수 없으므로 겹치는 단어 제거
        word2id[token] = i
        id2word[i] = token

    return word2id, id2word

def GenerateTrainData(tokens:list, word2id:dict, window:int) -> tuple[numpy.array, numpy.array]:

    x = []
    y = []
    n_tokens = len(tokens)
    
    for i in range(n_tokens):
        idx = concat(
            range(max(0, i - window), i), 
            range(i, min(n_tokens, i + window + 1))
        )
        for j in idx:
            if i == j: continue
            x.append(one_hot_encode(word2id[tokens[i]], len(word2id)))
            y.append(one_hot_encode(word2id[tokens[j]], len(word2id)))
    
    return numpy.asarray(x), numpy.asarray(y)

def concat(*iterables):
    for iterable in iterables:
        yield from iterable 

def one_hot_encode(id, vocab_size):
    res = [0] * vocab_size
    res[id] = 1
    return res