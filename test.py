import cream_beta as cream
import Csys
import numpy



import os

Csys.clear()

net = cream.Network([2,2,1],function=cream.sigmoid,learning_rate=0.3)
print(net.weights)

'''open(f'{os.getcwd()}\\visualizer\\data\\weights.dat','w',encoding='utf-8').write(str(net.weights))
cream.visualizer.run(f'{os.getcwd()}\\visualizer\\data\\weights.dat')'''

net.train(5, cream.dataset.XOR,10000)

Csys.stop()