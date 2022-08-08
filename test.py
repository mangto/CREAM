import cream_beta as cream
import Csys
 

import os

Csys.clear()


net = cream.snn([2,5,2],function=cream.ReLU,LearningRate=0.3)
net.train(3, cream.dataset.Reverse,EndCost=0.0000000001)

while True:
    input_ = str(input(" >>> "))
    input_ = input_.split(" ")
    if (len(input_) == 2 and input_[0] in ["1", "0"] and input_[1] in ["1", "0"]):
        input_ = [int(input_[0]),int(input_[1])]
        net.forwardfeed(input_)
        print(net.activations[-1])