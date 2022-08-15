import cream_beta as cream
import SNN as cream
import tool.Csys as Csys
import tool.progress_bar as pb


import numpy, random, time

Csys.clear()

network = cream.snn([2,4,1],cream.sigmoid)
print(network.biases)