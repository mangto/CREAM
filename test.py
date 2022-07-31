import cream_beta as cream
import Csys

import os

Csys.clear()

net = cream.Network([2,2,2,2,2],function=cream.sigmoid)
print(net.weights)

open(f'{os.getcwd()}\\visualizer\\data\\weights.dat','w',encoding='utf-8').write(str(net.weights))
cream.visualizer.run(f'{os.getcwd()}\\visualizer\\data\\weights.dat')

net.train(5, cream.dataset.Reverse,1)

Csys.stop()