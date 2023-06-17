from cream.functions.nlp import *

# __repr__ & __str__ function is for convinence of developing

class trie:
    def __init__(self) -> None:

        '''
        Very Simple Trie Node!
        * .child :dict = {'key1':trie, 'key2':trie}
        * .EOW :bool = True/False (Default: False)
        * .FailureLink :trie = None
        * .possibility :float = 0.0
        '''

        self.child :dict = {}
        self.EOW :bool = False # end of word
        self.appearance : int = 0
        self.output = []

        return

    def __str__(self) -> str:
        return '\033[95m' + str(self.appearance) + '|' + '\033[0m' + str(self.child)
    
    def __repr__(self) -> str:
        return '\033[95m' + str(self.appearance) + '|' + '\033[0m' + str(self.child)

class possibility:
    def __init__(self, possibility:float=0.0, next:int=0, tokens:list=[], TotalLen:int=0):
        self.possibility = possibility
        self.next = next
        self.tokens = tokens
        self.TotalLen = TotalLen

    def __call__(self):
        return self.possibility
    
    def __str__(self) -> str:
        return str(self.tokens)
    
    def __repr__(self) -> str:
        return '\033[95m' + str(round(self.possibility, 5)) + '|' + '\033[0m' + str(self.tokens)
    
    def __lt__(self, other):
         return self.possibility < other.possibility
    
class ahocorasick:
    def __init__(self) -> None:
        '''
        Aho Corasick Alogrithm with Possibility
        (This algorithm is for korean)
        '''

        self.root = trie()

        return
    
    def __str__(self) -> str:
        return str(self.root)
    
    def __repr__(self) -> str:
        return str(self.root)

    def AddLink(self, key:str):
        assert type(key) == str, 'wrong key type, must be string'

        node = self.root
        node.appearance += 1
        splited = SplitString(key)

        for char in splited:
            node :trie = node.child.setdefault(char, trie()) # find existing trie, unless exist -> new trie
            node.appearance += 1

        node.EOW = True
        node.output.append(key)
        
        return
    
    def AddLinks(self, keys:list[str]):
        for key in keys: self.AddLink(key)
        return
    
    def GetPossibility(self, key:str) -> dict[str:float]:
        splited = SplitString(key)
        node = self.root
        PrvApp = node.appearance # Previous Appearance
        possibility = -1.0
        chars = ''
        possibilities = {}
        index = []

        for i, char in enumerate(splited):
            chars += char
            node :trie = node.child.setdefault(char, None)
            if (node == None):
                # possibilities[chars] = 0.0
                # index.append(len(chars))
                return possibilities, index

            if (node.EOW):
                possibilities[chars] = possibility
                index.append(len(chars))

            App = node.appearance
            possibility *= App/PrvApp
            # if (i == 0): possibility = -1.0
            PrvApp = App

        if (possibilities == {}):
            possibilities[chars] = possibility
            index.append(len(chars))

        return possibilities, index
    
    def analyze(self, key:str) -> dict:
        '''
        Analyze Token
        
        <return value structure>
        {'token': [index, next_index, possibility], ...}
        '''


        splited = [''.join(SplitString(key))]
        indexes = [0]
        keys = {}
        new_splited = None
        new_index = None

        while new_splited != []:

            new_splited = []
            new_index = []

            for n, split in enumerate(splited):

                prv_index = indexes[n]
                possibilities, index = self.GetPossibility(split)

                for j, i in enumerate(index):
                    k = split[:i]
                    newi = prv_index+index[j] # new index
                    new = split[i:]

                    ci = newi-len(k)
                    if (ci not in keys): keys[ci] = [(newi, possibilities[k])]
                    else: keys[ci].append((newi, possibilities[k]))

                    if (new != '' and new not in keys):
                        new_splited.append(new)
                        new_index.append(newi)

            splited = new_splited
            indexes = new_index



        return keys

    def tokenizing(self, key:str, returner=max) -> list:
        analyzed = self.analyze(key)
        splited = ''.join(SplitString(key))

        poses = [possibility(inform[1], inform[0], [splited[:inform[0]]], inform[0]) for inform in analyzed.get(0, [])]
        exist = []

        for i, char in enumerate(splited[1:]):
            i += 1
            for pos in poses:
                if (pos.next == i):
                    news = analyzed.get(i, [])
                    poses.remove(pos)
                    prv_tokens = pos.tokens
                    
                    for inform in news:
                        tokens = prv_tokens+[splited[i:inform[0]]]
                        if (tokens not in exist):
                            exist.append(tokens)
                            poses.append(possibility(inform[1], inform[0], tokens, inform[0]))

        poses = [pos for pos in poses if pos.TotalLen == len(splited)]
        poses.sort()

        return returner(poses).tokens
    
    def tokenize(self, key:str) -> list:
        tokenized = self.tokenizing(key)
        result = []
        for token in tokenized:
            token = token if token[0] not in JUNGSUNG_LIST else 'ㅇ' + token
            result.append(join_jamos(token))

        return result