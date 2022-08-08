import cream_beta as cream
import Csys


import os

Csys.clear()

net = cream.snn([2,50,1],function=cream.sigmoid,LearningRate=5)
net.train(1, cream.dataset.XOR,EndCost=0)