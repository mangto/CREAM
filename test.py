import cream_beta as cream
import tool.Csys as Csys
import tool.progress_bar as pb
import numpy, random, time
from threading import Thread

Csys.clear()
 
network = cream.snn([2,5,1],cream.sigmoid,0.3)
print(network)

Csys.stop()
network.train(1,cream.dataset.XOR,20000)


while True:
    input_ = str(input(" >>> "))
    input_ = input_.split(" ")
    if (len(input_) == 2 and input_[0] in ["1", "0"] and input_[1] in ["1", "0"]):
        input_ = [int(input_[0]),int(input_[1])]
        network.forwardfeed(input_)
        print(network.activations[-1]) 